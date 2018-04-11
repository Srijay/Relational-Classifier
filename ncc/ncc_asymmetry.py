import os, sys, math, logging, argparse
import numpy as np
import cPickle as pickle
import matplotlib
import tensorflow as tf

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord, Phrase

# python ncc_asymmetry.py --indir /mnt/infoback/data/soumen/ncc/glove300 --phrases /data/srijayd/local_data/f_r1_r2_th/r1/r1_literals.csv --outdir /data/srijayd

class NccAsymmetry(NonComp):
    def __init__(self, flags):
        logging.info('Begin %s.__init__', NccAsymmetry.__name__)
        super(NccAsymmetry, self).__init__(flags)
        self.loadPhraseInstancesNcCsv(flags.phrases)
	self.makeFoldsCompounds()
	self.buildAsymmetryLosses()
        logging.info('End %s.__init__', NccAsymmetry.__name__)

    def diagnostics(self, msg, wpv):
        norms1 = np.linalg.norm(wpv, axis=1)
        avgNorm1 = np.average(norms1)
        print msg, avgNorm1

    def makeFoldsCompounds(self): # To make train and test folds
	ind = list(xrange(self.numInst))
	self.deterministicShuffle(ind)
	train = 0.5
	test = 0.5
	splitindextrain = int(math.floor(train*self.numInst))
	le = ind[:splitindextrain]
	ap = ind[splitindextrain:]
	le,ap = list(le),list(ap)
	print "here length of train and test are "
	print len(le)
	print len(ap)
	self.clW1, self.clW2, self.clW12, self.cll1, self.cll2, self.clY = [self.allWid1[x] for x in le], [self.allWid2[x] for x in le], [self.allWid12[x] for x in le], [self.literal1[x] for x in le], [self.literal2[x] for x in le], [self.allY[x] for x in le]
	self.caW1, self.caW2, self.caW12, self.cal1, self.cal2, self.caY = [self.allWid1[x] for x in ap], [self.allWid2[x] for x in ap], [self.allWid12[x] for x in ap], [self.literal1[x] for x in ap], [self.literal2[x] for x in ap], [self.allY[x] for x in ap]
	print "Folding Finished"

    def fun(self,a):
    	return (2*tf.sigmoid(a)-1)

    def buildAsymmetryLosses(self):

        self.w1 = tf.placeholder(tf.int32, [None])
        self.w2 = tf.placeholder(tf.int32, [None])
        self.w12 = tf.placeholder(tf.int32, [None])
	self.g1_real = tf.placeholder(tf.float32, [None])
	self.g2_real = tf.placeholder(tf.float32, [None])

	w1v = tf.nn.embedding_lookup(self.embedstf, self.w1)
        w2v = tf.nn.embedding_lookup(self.embedstf, self.w2) 
	w12v = tf.nn.embedding_lookup(self.embedstf, self.w12) 


	self.c1 = tf.Variable(1.0)
	self.b1 = tf.Variable(0.0)
	self.c2 = tf.Variable(1.0)
	self.b2 = tf.Variable(0.0)

	w1v_norm = tf.sqrt(tf.reduce_sum(w1v*w1v,1))
	w2v_norm = tf.sqrt(tf.reduce_sum(w2v*w2v,1))
	w12v_norm = tf.sqrt(tf.reduce_sum(w12v*w12v,1))

	self.g1_pred = tf.map_fn(self.fun,(self.c1*tf.reduce_sum(w1v*w12v,1))/1 + self.b1)
	self.g2_pred = tf.map_fn(self.fun,(self.c2*tf.reduce_sum(w2v*w12v,1))/1 + self.b2)
	
	#self.g1_pred = self.c1*tf.reduce_sum(w1v*w12v,1) + self.b1
	#self.g2_pred = self.c2*tf.reduce_sum(w2v*w12v,1) + self.b2

	self.loss1 = abs((self.g1_pred - self.g1_real))
	self.loss2 = abs((self.g2_pred - self.g2_real))

	self.netloss1 = tf.reduce_sum(self.loss1)
	self.netloss2 = tf.reduce_sum(self.loss2)

	print "Here ",self.g1_pred.get_shape()
	print "Here ",w1v.get_shape()

	est12 = tf.transpose(self.g1_pred * tf.transpose(w1v) + self.g2_pred * tf.transpose(w2v))
	est12norm = tf.sqrt(tf.reduce_sum(est12*est12,1))
	dot12 = tf.reduce_sum(tf.mul(est12, w12v),1)

	vec12est12norm = w12v_norm * est12norm
        self.cos12 = dot12 / vec12est12norm

	#self.totalLoss = pow(self.g1_pred,1) + pow(self.g2_pred,1)

	sgd = tf.train.AdagradOptimizer(0.7)
	self.trainop1 = sgd.minimize(self.loss1, var_list=[self.c1,self.b1])
	self.trainop2 = sgd.minimize(self.loss2, var_list=[self.c2,self.b2])
        self.initop = tf.initialize_all_variables()

    def getRandomCompMiniBatch(self, batchSize=20):

        sample = np.random.randint(0, len(self.clY), batchSize)
        sWid1 = [self.clW1[x] for x in sample]
        sWid2 = [self.clW2[x] for x in sample]
        sWid12 = [self.clW12[x] for x in sample]
        sLit1 = [self.cll1[x] for x in sample]
	sLit2 = [self.cll2[x] for x in sample]
        return sWid1, sWid2, sWid12, sLit1, sLit2


    def doTrain(self, sess, maxIters=30000):

	print "Welcome to training"

	print "Training for literal 1 Started"
        sess.run(self.initop, feed_dict={self.place: self.embeds})
	for xiter in xrange(maxIters):
		wid1, wid2, wid12, lit1, lit2 = self.getRandomCompMiniBatch()
		_, trloss = sess.run([self.trainop1, self.netloss1],
                                            feed_dict={self.w1: wid1,
                                                       self.w2: wid2,
						       self.w12: wid12,
                                                       self.g1_real: lit1
                                                       })
            	if xiter % 100 == 0:
                	teloss = sess.run(self.netloss1,
                                             	     feed_dict={self.w1: self.caW1,
                                                        	self.w2: self.caW2,
						       		self.w12: self.caW12,
                                                        	self.g1_real: self.cal1
                                                        	})
                	logging.info("CascadeModel xiter=%d trloss=%g teloss=%g",xiter, trloss, teloss)

	print "Training for literal 2 Started"
        #sess2.run(self.initop, feed_dict={self.place: self.embeds})
	for xiter in xrange(maxIters):
		wid1, wid2, wid12, lit1, lit2 = self.getRandomCompMiniBatch()
		_, trloss = sess.run([self.trainop2, self.netloss2],
                                            feed_dict={self.w1: wid1,
                                                       self.w2: wid2,
						       self.w12: wid12,
                                                       self.g2_real: lit1
                                                       })
            	if xiter % 100 == 0:
                	teloss = sess.run(self.netloss2,
                                             	     feed_dict={self.w1: self.caW1,
                                                        	self.w2: self.caW2,
						       		self.w12: self.caW12,
                                                        	self.g2_real: self.cal2
                                                        	})
                	logging.info("CascadeModel xiter=%d trloss=%g teloss=%g",xiter, trloss, teloss)

    def moveTrainToTestFold(self):
        """Empties training fold into test fold."""
        logging.warn("Moving %d instances from train to test", len(self.clY))
        self.caW1.extend(self.clW1)
        self.caW2.extend(self.clW2)
        self.caW12.extend(self.clW12)
	self.cal1.extend(self.cll1)
	self.cal2.extend(self.cll2)
        self.caY.extend(self.clY)
        del self.clW1[:]
        del self.clW2[:]
        del self.clW12[:]
	del self.cll1[:]
	del self.cll2[:]
        del self.clY[:]
        logging.warn("Updated train=%d test=%d folds", len(self.clY), len(self.caY))

    def doEval(self,sess):
    	print "Welcome to Eval"

        if self.flags.alltest:
            self.moveTrainToTestFold()

	finalscores,c1,b1,c2,b2 = sess.run([self.cos12,self.c1,self.b1,self.c2,self.b2],
							 feed_dict={self.w1: self.caW1,
                                                         self.w2: self.caW2,
						       	 self.w12: self.caW12,
                                                         self.g1_real: self.cal1,
							 self.g2_real: self.cal2
                                                        })

	print "final parameters are"
	print c1
	print b1
	print c2
	print b2

        # Larger loss thend to non-compositional
        xToYs = dict()
        for ix in xrange(len(self.caW12)):
            xToYs[','.join((self.vocabArray[self.allWid1[ix]].word,
                            self.vocabArray[self.allWid2[ix]].word))] = \
                [self.caY[ix], 1.0 - finalscores[ix]]
        outPicklePath = os.path.join(self.flags.outdir, 'linsum.pickle')
        with open(outPicklePath, 'wb') as outPickleFile:
            pickle.dump(xToYs, outPickleFile)
	nccEval = EvalRecallPrecisionF1(sortOrderDecr=True)
        # Most compositional first.
        res = nccEval.doEvalDict(xToYs)
        for k, v in res.iteritems():
            if k != 'recall' and k != 'precision':
                print >> sys.stderr, k, '-->', v
                logging.info("result %s %g", k, v)
        _, ax = plt.subplots()
        nccEval.plotRecallPrecision(ax, res['recall'], res['precision'], '', 'k')
        plt.show()

	

    def runMockPerfect(self):
        """Sets test score = gold score to check eval code."""
        xToYs = dict()
        for ix in xrange(len(self.allY)):
            xToYs[','.join((self.vocabArray[self.allWid1[ix]].word,
                            self.vocabArray[self.allWid2[ix]].word))] = \
                [self.allY[ix], self.allY[ix]]
        outPicklePath = os.path.join(self.flags.outdir, 'perfect.pickle')
        with open(outPicklePath, 'wb') as outPickleFile:
            pickle.dump(xToYs, outPickleFile)

    def runMockRandom(self):
        """Sets test score = uniform[0,1]."""
        xToYs = dict()
        for ix in xrange(len(self.allY)):
            xToYs[','.join((self.vocabArray[self.allWid1[ix]].word,
                            self.vocabArray[self.allWid2[ix]].word))] = \
                [self.allY[ix], np.random.uniform(0, 1, 1)]
        outPicklePath = os.path.join(self.flags.outdir, 'random.pickle')
        with open(outPicklePath, 'wb') as outPickleFile:
            pickle.dump(xToYs, outPickleFile)

def mainAsymmetryModel(flags):
    with tf.Session() as sess:
	print "Welcome to Asymmetric Gating Model"
        nc = NccAsymmetry(flags)
        nc.doTrain(sess)
        nc.doEval(sess)

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True, type=str,
                        help='/path/to/vocab/and/embeddings/')
    parser.add_argument("--phrases", required=True, type=str,
                        help='/path/to/labeled/phrases.csv')
    parser.add_argument("--outdir", required=True, type=str,
                        help='/path/to/save/outputs/')
    parser.add_argument("--unit", required=False, action='store_true',
                        help="Normalize embeddings to unit L2 length?")
    parser.add_argument("--alltest", nargs='?', const=True, default=False, type=bool,
                        help='Apply model to all instances?')
    args = parser.parse_args()
    mainAsymmetryModel(args)


