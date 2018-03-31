# Implementation of distributional anomaly algorithms:
# Dima's embedding based test
# https://aclweb.org/anthology/D/D15/D15-1188.pdf

#python ncc_dima_tf.py --glove /data/srijayd/local_data/glove_embeddings/ --phrases /data/srijayd/local_data/f_r1_r2_th/f/f.csv --model /data/srijayd/ncc/src/ncc/phrases_farahmand/dima_fle/ --train True --alltest True

import os, sys, math, logging, argparse
import numpy as np
import cPickle as pickle

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, logistic, spearmanr
from enum import Enum

import tensorflow as tf

from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord, Phrase


class NccInTestFold(Enum):
    """DATs are trained on compositional phrases only.
    Move this fraction of NCCs to the test fold."""
    nothing = 1
    half = 2
    all = 3


class NccDiscrepancy(NonComp):
    """Read noncomp labeled data e.g. Farahmand. Retain only comp instances.
    Train on train fold using "interaction" based loss of Yazdani+.
    Save TF model. Apply on test fold and plot correlation.
    """

    regularizer = 0.0001
    '''Test numbers are quite sensitive to the choice of regularizer.
    Regularizer used for Yazdano, not Dima.'''

    def __init__(self, flags, thresh=0.5):
        logging.info('Begin %s.__init__', NccDiscrepancy.__name__)
        super(NccDiscrepancy, self).__init__(flags)
        self.thresh = thresh
        self.phrasesPath = flags.phrases
        self.phrasesDir = os.path.dirname(os.path.realpath(self.phrasesPath))
        logging.info('phrasesDir=%s', self.phrasesDir)
        # first load CSV and split train test
        self.loadPhraseInstancesNcCsv(self.phrasesPath)
        self.selectCompPhrases()
        # then replace training set with unlabeled data if flag set
	
        self.buildLossDima8()
        logging.info('End %s.__init__', NccDiscrepancy.__name__)


    def selectCompPhrases(self, nonCompInTestFold=NccInTestFold.all):
        """Samples only compositional (gold noncomp score <= self.thresh)
        labeled instances for train fold.  Samples rest of compositional instances
        for test fold, and optionally adds noncomp instances."""
        ind = []
        for inst in xrange(self.numInst):
            if self.allY[inst] < self.thresh: ind.append(inst)
        logging.info("%d of %d instances comp", len(ind), self.numInst)
        self.deterministicShuffle(ind)
        train = 0.7 #As did by Dima
	dev = 0.1
	test = 0.2
	splitindextrain = int(math.floor(train*len(ind)))
	splitindexdev = int(math.floor(dev*len(ind)))
	le = ind[:splitindextrain]
	de = ind[splitindextrain:splitindextrain + splitindexdev]
	ap = ind[splitindextrain + splitindexdev:]
        logging.info("Split %d comp into %d %d", len(ind), len(le), len(ap))
	'''
        xfer = []  # noncomp indices
        for inst in xrange(self.numInst):
            if self.allY[inst] >= self.thresh:
                xfer.append(inst)
        if nonCompInTestFold == NccInTestFold.all:
            ap.extend(xfer)
            logging.info("Padded with noncomp to le=%d ap=%d", len(le), len(ap))
        if nonCompInTestFold == NccInTestFold.half:
            self.deterministicShuffle(xfer)
            ap.extend(np.array_split(xfer, 2)[1])
            logging.info("Padded with noncomp to le=%d ap=%d", len(le), len(ap))
	'''
        self.clW1, self.clW2, self.clW12, self.clY = \
            [self.allWid1[x] for x in le], [self.allWid2[x] for x in le], \
            [self.allWid12[x] for x in le], [self.allY[x] for x in le]
	self.cdW1, self.cdW2, self.cdW12, self.cdY = \
	    [self.allWid1[x] for x in de], [self.allWid2[x] for x in de], \
	    [self.allWid12[x] for x in de], [self.allY[x] for x in de]
        self.caW1, self.caW2, self.caW12, self.caY = \
            [self.allWid1[x] for x in ap], [self.allWid2[x] for x in ap], \
            [self.allWid12[x] for x in ap], [self.allY[x] for x in ap]
        logging.info("%s finished with le=%d ap=%d",
                     self.selectCompPhrases.__name__,
                     len(self.clY), len(self.caY))

    def moveTrainToTestFold(self):
        """Empties training fold into test fold."""
        logging.warn("Moving %d instances from train to test", len(self.clY))
        self.caW1.extend(self.clW1)
        self.caW2.extend(self.clW2)
        self.caW12.extend(self.clW12)
        self.caY.extend(self.clY)
        del self.clW1[:]
        del self.clW2[:]
        del self.clW12[:]
        del self.clY[:]
        logging.warn("Updated train=%d test=%d folds", len(self.clY), len(self.caY))

    def getRandomCompMiniBatch(self, batchSize=50):
        """Like NonComp.getRandomMiniBatch, restricted to comp train fold."""
        sample = np.random.randint(0, len(self.clY), batchSize)
        sWid1 = [self.clW1[x] for x in sample]
        sWid2 = [self.clW2[x] for x in sample]
        sWid12 = [self.clW12[x] for x in sample]
        sY = [self.clY[x] for x in sample]
        return sWid1, sWid2, sWid12, sY

    def buildLossDima8(self):
        """Network 8 from Dima paper, roughly what Yazdani calls
        "linear projection".  Predicted phrase vector is w1v * M1 + w2v * M2.
        Here M1 and M2 are the model weights, two D times D matrices.  No
        nonlinearity is used.
        """
        self.w1p = tf.placeholder(tf.int32, [None])
        self.w2p = tf.placeholder(tf.int32, [None])
        self.w12p = tf.placeholder(tf.int32, [None])
        self.yp = tf.placeholder(tf.float32, [None])  # not used, should be 0 (comp)
        w1v = tf.nn.embedding_lookup(self.embedstf, self.w1p)
        w2v = tf.nn.embedding_lookup(self.embedstf, self.w2p)
        w12v = tf.nn.embedding_lookup(self.embedstf, self.w12p)
        matSize = self.dim * self.dim  #numDim->dim for glove
        stddev = 1. / math.sqrt(matSize)
        M1 = self.buildVariable([self.dim, self.dim], stddev, 'M1') #numDim->dim for glove
        M2 = self.buildVariable([self.dim, self.dim], stddev, 'M2') #numDim->dim for glove
        predPhrase = tf.matmul(w1v, M1) + tf.matmul(w2v, M2)
        diffs = tf.sub(predPhrase, w12v)
        diffs2 = tf.mul(diffs, diffs)
        self.rowLosses = tf.reduce_sum(diffs2, 1)
        self.loss = tf.reduce_mean(self.rowLosses)  # without regularization
	reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  	reg_constant = 0.01  # Choose an appropriate one.
  	self.lossreg = self.loss #+ reg_constant * sum(reg_losses) # regularization added
        #self.lossreg = self.loss
        sgd = tf.train.AdagradOptimizer(.9)
	#sgd = tf.train.FtrlOptimizer(.8,l1_regularization_strength=1.0,l2_regularization_strength=1.0)
        self.trainop = sgd.minimize(self.lossreg, var_list=[M1, M2])
        self.initop = tf.initialize_all_variables()
        self.saver = tf.train.Saver(tf.trainable_variables())
        logging.info("Built Dima network 8.")

    def doTrain(self, sess, maxIters=1000000):
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel.tf')
        if os.path.exists(tfModelPath):
            self.saver.restore(sess, tfModelPath)
            logging.info("Warmstart from %s", tfModelPath)
	    print "Welcome to training"
        sess.run(self.initop, feed_dict={self.place: self.embeds})
	print "Welcome to training"
        #INIT sess.run(tf.initialize_variables([self.embedstf]))
	prevAccuracy = 0
	count = 0
	print "train set length - ",len(self.clW1)
	print "dev set length - ",len(self.cdW1)
	print "test set length - ",len(self.caW1)
        for xiter in xrange(maxIters):
            wid1, wid2, wid12, y = self.getRandomCompMiniBatch()
            _, trloss, trlossreg,rowLosses = sess.run([self.trainop, self.loss, self.lossreg,self.rowLosses],
                                            feed_dict={self.w1p: wid1,
                                                       self.w2p: wid2,
                                                       self.w12p: wid12,
                                                       self.yp: y
                                                       })
            if xiter % 1000 == 0:
                teloss, telossreg = sess.run([self.loss, self.lossreg],
                                             feed_dict={self.w1p: self.caW1,
                                                        self.w2p: self.caW2,
                                                        self.w12p: self.caW12,
                                                        self.yp: self.caY
                                                        })
                logging.info("xiter=%d trloss=%g trlossreg=%g teloss=%g telossreg=%g",
                            xiter, trloss, trlossreg, teloss, telossreg)
	
	  #Code for Early stopping for Dima43
		
	    currAccuracy = self.getAccuracy(sess,self.cdW1,self.cdW2,self.cdW12,self.cdY)
	    print "Accuracy is ",currAccuracy
	    exit(0)
	    if(currAccuracy > prevAccuracy):
	   	 prevAccuracy = currAccuracy
		 self.saver.save(sess, tfModelPath)	
		 logging.info("")
		 logging.info("Validation Accuracy => %g",prevAccuracy)
		 logging.info("")
		 count = 0
	    else:
		 count += 1
		 if(count > 10000):
		 	return	

    def calClass(self,sess,w1,w2,w12,y):
	rowloss = sess.run(self.rowLosses,feed_dict={self.w1p: [w1],self.w2p: [w2],self.w12p: [w12],self.w2p: [y]})
	sysScore = 2. * logistic.cdf(rowloss[0]) - 1.
	predClass = self.makeCompScore(sysScore)
	return predClass

    def getAccuracy(self,sess,w1s,w2s,w12s,ys):
	l = len(ys)
	count = 0
	for i in xrange(l):
		print "here ",self.calClass(sess,w1s[i],w2s[i],w12s[i],ys[i])," ",self.makeCompScore(ys[i])
		if(self.calClass(sess,w1s[i],w2s[i],w12s[i],ys[i]) == self.makeCompScore(ys[i])):
			count += 1
	return (count*1.0)/l

    def makeCompScore(self,x):
    	if(x <= 0.5):
		return 0
	else:
		return 1

    def doEval(self, sess, doPlot=True):
        """Runs on test or all folds.  Saves to a dict with key = 'mod,head'
        and values = gold label, system label.  Optionally plot recall-precision."""
	print "Welcome to Eval"
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel.tf')
        assert os.path.exists(tfModelPath), 'Cannot load ' + tfModelPath
        self.saver.restore(sess, tfModelPath)
        logging.info("Warmstart from %s", tfModelPath)

        #INIT sess.run(tf.initialize_variables([self.embedstf]))
        #sess.run(self.initop, feed_dict={self.place: self.embeds})
	sess.run(tf.initialize_variables([self.embedstf]),feed_dict={self.place: self.embeds})     #For pretrained word embeddings

        if self.flags.alltest:
            self.moveTrainToTestFold()

        _w1, _w2, _w12, _y = self.caW1, self.caW2, self.caW12, self.caY
	#_w1, _w2, _w12, _y = self.clW1, self.clW2, self.clW12, self.clY
        rowLosses = sess.run(self.rowLosses,
                             feed_dict={self.w1p: _w1,
                                        self.w2p: _w2,
                                        self.w12p: _w12,
                                        self.yp: _y,
                                        })
        # rowLosses are arbitrary nonnegative numbers; turn them into [0,1]
        # sysScores = rowLosses
        sysScores = [(2. * logistic.cdf(rl) - 1.) for rl in rowLosses]
	
	print "train accuracy is ",self.getAccuracy(sess,self.clW1,self.clW2,self.clW12,self.clY)
	print "test accuracy is ",self.getAccuracy(sess,self.caW1,self.caW2,self.caW12,self.caY) 
	
	exit(0)
        xToYs = dict()
        for ix in xrange(len(_y)):
            xToYs[','.join((self.tidToTerm[_w1[ix]],self.tidToTerm[_w2[ix]]))] = [_y[ix], sysScores[ix]] # self.vocabArray[_w1[ix]].word -> self.tidToTerm[_w1[ix]]
        outPicklePath = os.path.join(tfModelDirPath, 'pred.pickle')
        with open(outPicklePath, 'wb') as outPickleFile:
            pickle.dump(xToYs, outPickleFile)
        logging.info('Done %s', self.doEval.__name__)
        if not doPlot: return
        nccEval = EvalRecallPrecisionF1()
        res = nccEval.doEvalDict(xToYs)
        for k, v in res.iteritems():
            if k != 'recall' and k != 'precision':
                print >> sys.stderr, k, '-->', v
                logging.info("result %s %g", k, v)
        _, ax = plt.subplots()
        nccEval.plotRecallPrecision(ax, res['recall'], res['precision'], '', 'k')
        plt.show()


def mainDima(flags):
    with tf.Session() as sess:
        nc = NccDiscrepancy(flags)
	print "Welcome"
	print nc.flags.train
        if nc.flags.train: nc.doTrain(sess)
        nc.doEval(sess)


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", required=True, type=str,
                        help='/path/to/vocab/and/embeddings/') 
    parser.add_argument("--phrases", required=True, type=str,
                        help='/path/to/labeled/phrases.csv')
    parser.add_argument("--model", required=True, type=str,
                        help='/dir/holding/model/and/predictions')
    parser.add_argument("--train", nargs='?', const=True, default=False, type=bool,
                        help='Train using labeled instances?')
    parser.add_argument("--unsuper", nargs='?', const=True, default=False, type=bool,
                        help='Sample phrase alphabet for training?')
    parser.add_argument("--alltest", nargs='?', const=True, default=False, type=bool,
                        help='Apply model to all instances?')
    args = parser.parse_args()
    mainDima(args)







