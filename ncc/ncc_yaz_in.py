# Implementation of distributional anomaly algorithms:
# Yazdani et al.'s embedding based test
# http://anthology.aclweb.org/D/D15/D15-1201.pdf
# Dima's embedding based test
# https://aclweb.org/anthology/D/D15/D15-1188.pdf

# python ncc_yaz_in.py --indir /mnt/infoback/data/soumen/ncc/glove300 --phrases /data/srijayd/ncc/src/ncc/phrases_farahmand/farahmand_noncomp.csv --net yazdani --model /data/srijayd/ncc/src/ncc/phrases_farahmand/yazdani_fle/ --train True


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
    Regularizer used for Yazdani, not Dima.'''

    def __init__(self, flags, thresh=0.5):
        logging.info('Begin %s.__init__', NccDiscrepancy.__name__)
        super(NccDiscrepancy, self).__init__(flags)
        self.thresh = thresh
        self.phrasesPath = flags.phrases
        self.phrasesDir = os.path.dirname(os.path.realpath(self.phrasesPath))
        logging.info('phrasesDir=%s', self.phrasesDir)
        # first load CSV and split train test
        self.loadPhraseInstancesNcCsv(self.phrasesPath)
        self.trainableVarList = []
        if self.flags.net == 'dima':
            assert not flags.unsuper
            self.trainTestSplit()
            ##self.buildLossDima8()
            self.buildLossDimaDiscriminative()
        elif self.flags.net == 'salehi':
            assert not flags.unsuper
            self.trainTestSplit()
            self.buildLossSalehi()
        elif self.flags.net == 'sigyaz':
            assert not flags.unsuper
            self.trainTestSplit()
            self.buildLossYazdaniDiscriminative()
        elif self.flags.net == 'yazdani':
            self.selectCompPhrases()
            # then replace training set with unlabeled data if flag set
            if flags.unsuper:
                self.sampleUnlabeledTrainingPhrases()
            self.buildLossYazdani()
        logging.info('End %s.__init__', NccDiscrepancy.__name__)

    def sampleUnlabeledTrainingPhrases(self, minCount=100, numSamples=100000):
        """A vocab entry survives if it has minCount, and both constituent words
        are known to the vocabulary.  All these will be forced to have
        label=comp (0).  Sampling is deterministic.  Does not upset test set."""
        ind = []
        for ax in xrange(len(self.vocabArray)):
            alph = self.vocabArray[ax]
            if alph.kind != Phrase: continue
            if alph.count < minCount: continue
            words = alph.word.split('^')
            sMod, sHead = words[0], words[1]
            if sMod not in self.vocabMap: continue
            if sHead not in self.vocabMap: continue
            ind.append(ax)
        self.deterministicShuffle(ind, 43)
        ind = ind[:numSamples]
        logging.info("Sampled %d from %d alphabet entries.",
                     len(ind), len(self.vocabArray))
        # save the sample to a scratch file for debugging
        samplePath = os.path.join(self.flags.indir, 'unlabeled_sample.txt')
        with open(samplePath, 'w') as sampleFile:
            for ix in ind:
                sampleFile.write(self.vocabArray[ix].word + '\n')
        # TODO(soumenc) should we upset all* fields or only cl* fields?
        self.allWid1, self.allWid2, self.allWid12, self.allY = [], [], [], []
        for ix in ind:
            alph = self.vocabArray[ix]
            words = alph.word.split('^')
            sMod, sHead = words[0], words[1]
            if sMod not in self.vocabMap: continue
            if sHead not in self.vocabMap: continue
            self.allWid1.append(self.vocabMap[sMod])
            self.allWid2.append(self.vocabMap[sHead])
            self.allWid12.append(alph.wid)
            self.allY.append(0.)
        self.numInst = len(self.allY)
        self.clW1 = self.allWid1
        self.clW2 = self.allWid2
        self.clW12 = self.allWid12
        self.clY = self.allY
        logging.info("Samples train=%d test=%d", len(self.clY), len(self.caY))
        logging.flush()

    def selectCompPhrases(self, nonCompInTestFold=NccInTestFold.all):
        """Samples only compositional (gold noncomp score <= self.thresh)
        labeled instances for train fold.  Samples rest of compositional instances
        for test fold, and optionally adds noncomp instances."""
        ind = []
        for inst in xrange(self.numInst):
            if self.allY[inst] < self.thresh: ind.append(inst)
        logging.info("%d of %d instances comp", len(ind), self.numInst)
        self.deterministicShuffle(ind)
        le, ap = np.array_split(ind, 2)  # so few we do only two folds
        le, ap = le.tolist(), ap.tolist()  # so that we can append
        logging.info("Split %d comp into %d %d", len(ind), len(le), len(ap))
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
        self.clW1, self.clW2, self.clW12, self.clY = \
            [self.allWid1[x] for x in le], [self.allWid2[x] for x in le], \
            [self.allWid12[x] for x in le], [self.allY[x] for x in le]
        self.caW1, self.caW2, self.caW12, self.caY = \
            [self.allWid1[x] for x in ap], [self.allWid2[x] for x in ap], \
            [self.allWid12[x] for x in ap], [self.allY[x] for x in ap]
        logging.info("%s finished with le=%d ap=%d",
                     self.selectCompPhrases.__name__,
                     len(self.clY), len(self.caY))

    def moveTrainToTestFold(self):
        """Empties training fold into test fold."""
        if not hasattr(self, 'clY'):
            logging.warn('%s has nothing to move', self.moveTrainToTestFold.__name__)
            return
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

    def buildLossYazdani(self):
        """Interaction terms only, from Yazdani+ paper, with L1 regularization.
        Predicted phrase vector = vec(outer(w1v, w2v)) * theta
        Here w1v, w2v are vectors with D = self.numDim elements.
        outer(..) is their outer product, a D times D matrix.
        vec(..) flattens that to a D**2 dim vector.
        theta is the model, a D times D**2 matrix.
        """
        self.w1p = tf.placeholder(tf.int32, [None])
        self.w2p = tf.placeholder(tf.int32, [None])
        self.w12p = tf.placeholder(tf.int32, [None])
        self.yp = tf.placeholder(tf.float32, [None])  # not used
        w1v = tf.nn.embedding_lookup(self.embedstf, self.w1p)
        w2v = tf.nn.embedding_lookup(self.embedstf, self.w2p)
        w12v = tf.nn.embedding_lookup(self.embedstf, self.w12p)
        #w1ve = tf.expand_dims(w1v, 2)
        #w2ve = tf.expand_dims(w2v, 1)
	#print w1v.get_shape()
	#print w2v.get_shape()
	#sess1 = tf.Session()
	wcon=tf.concat(1,[w1v,w2v])
        #wcon2 = tf.expand_dims(wcon, 2)
        #wcon1 = tf.expand_dims(wcon, 1)
	#wcon	
	#w1vnp=sess.run(w1v)
	#w2vnp=sess.run(w2v)
	dimSquared = self.numDim * (self.numDim-1) /2
	res = tf.constant([])
	res=tf.reshape(res,[0,dimSquared])
	for k in range(0,50):
		#res=tf.constant([])
		l=tf.constant([])
		#print k
		for i in range(0,600):
			print "Here ",i
			for j in range(i+1,600):
				#print k[i]," ",k[j] 
				l = tf.concat(0,[l,[wcon[k][i]*wcon[k][j]]])
		#print l
		res = tf.concat(0,[res,[l]])
	
	
          #  l = tf.concat(0,[l,[a[i]*a[j]]])		
        #outer = tf.batch_matmul(wcon2, wcon1)
        #dimSquared = self.numDim * (self.numDim-1) /2
        stddev = 1. / math.sqrt(dimSquared)
        theta = self.buildVariable([dimSquared, self.numDim], stddev, name='theta')
        self.trainableVarList.append(theta)
        #flat = tf.squeeze(tf.reshape(outer, [-1, dimSquared, 1]))
	#print flat.get_shape()
	#flat=tf.convert_to_tensor(res)
        predV12 = tf.matmul(res, theta)
        logging.debug('v12 %s predV12 %s', str(w12v), str(predV12))
        diffs = tf.sub(predV12, w12v)
        diffs2 = tf.mul(diffs, diffs)
        self.rowDiscrepancies = tf.reduce_sum(diffs2, 1)
        self.rowLosses = self.rowDiscrepancies
        self.loss = tf.reduce_mean(self.rowLosses)  # without regularization
        l1normTheta = tf.reduce_sum(tf.abs(theta))
        self.lossreg = self.loss + self.regularizer * l1normTheta  # with
        sgd = tf.train.AdagradOptimizer(.3)
        #sgd = tf.train.GradientDescentOptimizer(.1)
        self.trainop = sgd.minimize(self.lossreg, var_list=[theta])
        self.saver = tf.train.Saver(tf.trainable_variables())
        logging.info("Built Yazdani network.")

    def buildLossYazdaniDiscriminative(self):
        '''Like Yazdani but runs on all instances, not just comp instances;
        also uses cosine, not L2 discrepancy, and applies tuned sigmoid to
        obtain final noncomp score.'''
        self.w1p = tf.placeholder(tf.int32, [None])
        self.w2p = tf.placeholder(tf.int32, [None])
        self.w12p = tf.placeholder(tf.int32, [None])
        self.yp = tf.placeholder(tf.float32, [None])
        w1v = tf.nn.embedding_lookup(self.embedstf, self.w1p)
        w2v = tf.nn.embedding_lookup(self.embedstf, self.w2p)
        w12v = tf.nn.embedding_lookup(self.embedstf, self.w12p)
        w12vNorm = tf.sqrt(tf.reduce_sum(tf.mul(w12v, w12v), 1))
        w1ve = tf.expand_dims(w1v, 2)
        w2ve = tf.expand_dims(w2v, 1)
        outer = tf.batch_matmul(w1ve, w2ve)
        dimSquared = self.numDim * self.numDim
        stddev = 1. / math.sqrt(dimSquared)
        theta = self.buildVariable([dimSquared, self.numDim], stddev, name='theta')
        l1normTheta = tf.reduce_sum(tf.abs(theta))
        self.trainableVarList.append(theta)
        flat = tf.squeeze(tf.reshape(outer, [-1, dimSquared, 1]))
        predV12 = tf.matmul(flat, theta)
        logging.debug('v12 %s predV12 %s', str(w12v), str(predV12))
        # End cut and paste
        predV12norm = tf.sqrt(tf.reduce_sum(tf.mul(predV12, predV12), 1))
        dot_w12_pred12 = tf.reduce_sum(tf.mul(w12v, predV12), 1)
        cos_w12_pred12 = dot_w12_pred12 / w12vNorm / predV12norm
        slope = tf.Variable(name='slope', initial_value=-1, dtype=tf.float32)
        self.trainableVarList.append(slope)
        offset = tf.Variable(name='slope', initial_value=0, dtype=tf.float32)
        self.trainableVarList.append(offset)
        self.rowDiscrepancies = tf.nn.sigmoid((cos_w12_pred12 - offset) * slope)
        self.rowLosses = tf.abs(tf.sub(self.rowDiscrepancies, self.yp))
        self.loss = tf.reduce_mean(self.rowLosses)
        self.lossreg = self.loss + self.regularizer * l1normTheta
        sgd = tf.train.AdagradOptimizer(.3)
        self.trainop = sgd.minimize(self.lossreg, var_list=self.trainableVarList)
        self.saver = tf.train.Saver(tf.trainable_variables())

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
        matSize = self.numDim * self.numDim
        stddev = 1. / math.sqrt(matSize)
        # FIXME
        #M1 = self.buildVariable([self.numDim, self.numDim], stddev, 'M1')
        #M2 = self.buildVariable([self.numDim, self.numDim], stddev, 'M2')
        M1 = tf.Variable(initial_value = np.identity(self.numDim)/2., dtype=tf.float32)
        M2 = tf.Variable(initial_value = np.identity(self.numDim)/2., dtype=tf.float32)
        self.trainableVarList.extend([M1, M2])
        predPhrase = tf.matmul(w1v, M1) + tf.matmul(w2v, M2)
        diffs = tf.sub(predPhrase, w12v)
        diffs2 = tf.mul(diffs, diffs)
        self.rowLosses = tf.reduce_sum(diffs2, 1)
        self.loss = tf.reduce_mean(self.rowLosses)  # without regularization
        self.lossreg = self.loss
        sgd = tf.train.AdagradOptimizer(.3)
        self.trainop = sgd.minimize(self.lossreg, var_list=[M1, M2])
        self.saver = tf.train.Saver(tf.trainable_variables())
        logging.info("Built Dima8 network.")

    def buildLossDimaDiscriminative(self):
        """Dima8, but training with both comp and noncomp instances."""
        self.w1p = tf.placeholder(tf.int32, [None])
        self.w2p = tf.placeholder(tf.int32, [None])
        self.w12p = tf.placeholder(tf.int32, [None])
        self.yp = tf.placeholder(tf.float32, [None])
        w1v = tf.nn.embedding_lookup(self.embedstf, self.w1p)
        w2v = tf.nn.embedding_lookup(self.embedstf, self.w2p)
        w12v = tf.nn.embedding_lookup(self.embedstf, self.w12p)
        matSize = self.numDim * self.numDim
        stddev = 1. / math.sqrt(matSize)
        # M1 = self.buildVariable([self.numDim, self.numDim], stddev, 'M1')
        M1 = tf.Variable(name='M1', initial_value = np.identity(self.numDim)/2., dtype=tf.float32)
        # M2 = self.buildVariable([self.numDim, self.numDim], stddev, 'M2')
        M2 = tf.Variable(name='M2', initial_value = np.identity(self.numDim)/2., dtype=tf.float32)
        self.trainableVarList.extend([M1, M2])
        predPhrase = tf.matmul(w1v, M1) + tf.matmul(w2v, M2)
        rowDots = tf.reduce_sum(tf.mul(w12v, predPhrase), 1)
        w12vNorm2 = tf.sqrt(tf.reduce_sum(tf.mul(w12v, w12v), 1))
        predPhraseNorm2 = tf.sqrt(tf.reduce_sum(tf.mul(predPhrase, predPhrase), 1))
        rowCosines = tf.div(tf.div(rowDots, w12vNorm2), predPhraseNorm2)
        self.rowDiscrepancies = tf.squeeze(tf.sub(tf.ones_like(rowCosines), rowCosines))
        self.rowLosses = tf.abs(tf.sub(self.rowDiscrepancies, self.yp))
        self.loss = tf.reduce_mean(self.rowLosses)
        self.lossreg = self.loss
        sgd = tf.train.AdagradOptimizer(.3)
        self.trainop = sgd.minimize(self.lossreg, var_list=self.trainableVarList)
        self.saver = tf.train.Saver(tf.trainable_variables())
        logging.info("Built Dima8 discriminative network.")

    def buildLossSalehi(self):
        '''Tweak on Salehi. Compute rowComp as in Salehi with one tuned
        blender, then subject to a sigmoid with tuned slope and offset.'''
        self.w1p = tf.placeholder(tf.int32, [None])
        self.w2p = tf.placeholder(tf.int32, [None])
        self.w12p = tf.placeholder(tf.int32, [None])
        self.yp = tf.placeholder(tf.float32, [None])
        v1 = tf.nn.embedding_lookup(self.embedstf, self.w1p)
        v1norm = tf.sqrt(tf.reduce_sum(tf.mul(v1, v1), 1))
        v2 = tf.nn.embedding_lookup(self.embedstf, self.w2p)
        v2norm = tf.sqrt(tf.reduce_sum(tf.mul(v2, v2), 1))
        v12 = tf.nn.embedding_lookup(self.embedstf, self.w12p)
        v12norm = tf.sqrt(tf.reduce_sum(tf.mul(v12, v12), 1))
        dot_v1_v12 = tf.reduce_sum(tf.mul(v1, v12), 1)
        cos_v1_v12 = dot_v1_v12 / v1norm / v12norm
        dot_v2_v12 = tf.reduce_sum(tf.mul(v2, v12), 1)
        cos_v2_v12 = dot_v2_v12 / v2norm / v12norm
        bal = tf.Variable(name='bal', initial_value = 0, dtype=tf.float32)
        self.trainableVarList.append(bal)
        sigbal = tf.nn.sigmoid(bal)
        rowComps = cos_v1_v12 * sigbal + cos_v2_v12 * (1. - sigbal)
        slope = tf.Variable(name='slope', initial_value=-1, dtype=tf.float32)
        self.trainableVarList.append(slope)
        offset = tf.Variable(name='slope', initial_value=0, dtype=tf.float32)
        self.trainableVarList.append(offset)
        self.rowDiscrepancies = tf.nn.sigmoid(slope * (rowComps - offset))
        self.rowLosses = tf.abs(tf.sub(self.rowDiscrepancies, self.yp))
        self.loss = tf.reduce_mean(self.rowLosses)
        self.lossreg = self.loss
        sgd = tf.train.AdagradOptimizer(.3)
        self.trainop = sgd.minimize(self.lossreg, var_list=self.trainableVarList)
        self.saver = tf.train.Saver(tf.trainable_variables())

    def doTrainDiscriminative(self, sess):
        """Training with both comp and noncomp instances."""
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel.tf')
        if self.flags.warm and os.path.exists(tfModelPath):
            self.saver.restore(sess, tfModelPath)
            logging.info("Warmstart from %s", tfModelPath)
        logging.info('Begin %s', self.doTrainDiscriminative.__name__)

        self.loadEmbeddingsTf(sess)
        sess.run(tf.initialize_all_variables(),
                 feed_dict={self.place: self.embeds})
        sess.run(tf.initialize_variables(self.trainableVarList))

        saveGap = min(self.flags.batches/10, 100)
        for xiter in xrange(self.flags.batches):
            wid1, wid2, wid12, y = self.getRandomMiniBatch(500)
            _, trloss, trlossreg = sess.run([self.trainop, self.loss, self.lossreg],
                                            feed_dict={self.w1p: wid1,
                                                       self.w2p: wid2,
                                                       self.w12p: wid12,
                                                       self.yp: y })
            if xiter % saveGap == 0:
                teloss, telossreg = sess.run([self.loss, self.lossreg],
                                             feed_dict={self.w1p: self.teWid1,
                                                        self.w2p: self.teWid2,
                                                        self.w12p: self.teWid12,
                                                        self.yp: self.teY
                                                        })
                logging.info("xiter=%d trloss=%g trlossreg=%g teloss=%g telossreg=%g",
                             xiter, trloss, trlossreg, teloss, telossreg)

        logging.info('End %s', self.doTrainDiscriminative.__name__)
        self.saver.save(sess, tfModelPath)
        self.saveModelNumpy(sess)
        trDiscs = sess.run(self.rowDiscrepancies,
                          feed_dict={ self.w1p: self.trWid1,
                                      self.w2p: self.trWid2,
                                      self.w12p: self.trWid12,
                                      self.yp: self.trY, })
        teDiscs = sess.run(self.rowDiscrepancies,
                          feed_dict={ self.w1p: self.teWid1,
                                      self.w2p: self.teWid2,
                                      self.w12p: self.teWid12,
                                      self.yp: self.teY, })
        plt.plot(self.trY, trDiscs, 'o', self.teY, teDiscs, 'x')
        plt.show()
        # Output predictions on test fold.
        xToYs = dict()
        for ix in xrange(len(self.teY)):
            xToYs[','.join((self.vocabArray[self.teWid1[ix]].word, \
                            self.vocabArray[self.teWid2[ix]].word))] = [self.teY[ix], teDiscs[ix]]
        outPicklePath = os.path.join(tfModelDirPath, 'predictions.pickle')
        with open(outPicklePath, 'wb') as outPickleFile:
            pickle.dump(xToYs, outPickleFile)
        nccEval = EvalRecallPrecisionF1()
        res = nccEval.doEvalDict(xToYs)
        logging.info('After %s test_ktau %g',
                     self.doTrainDiscriminative.__name__,
                     res['ktau'])
        for k, v in res.iteritems():
            if k != 'recall' and k != 'precision':
                logging.debug("result %s %g", k, v)
        _, ax = plt.subplots()
        nccEval.plotRecallPrecision(ax, res['recall'], res['precision'], '', 'k')
        plt.show()

    def saveModelNumpy(self, sess):
        """Save trained model weights (not word embeddings) to pickled numpy matrices."""
        npModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
        npModelPath = os.path.join(npModelDirPath, "compmodel.numpickle")
        with open(npModelPath, 'wb') as npModelFile:
            pickle.dump(sess.run(self.trainableVarList), npModelFile)
        logging.info("Saved %s to %s", str(self.trainableVarList), npModelPath)

    def loadModelNumpy(self):
        """Load model weights from pickled numpy matrices.  To be kept in numpy
        matrices, not meant to be loaded into tf variables. Therefore no session arg."""
        npModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
        npModelPath = os.path.join(npModelDirPath, "compmodel.numpickle")
        with open(npModelPath, 'rb') as npModelFile:
            savedVarList = pickle.load(npModelFile)
        logging.info("Loaded %d numpy vars from %s", len(savedVarList), npModelPath)
        return savedVarList

    def doTrain(self, sess):
        """Training with only comp instances."""
        logging.info("Train begin.")
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel.tf')
        if self.flags.warm and os.path.exists(tfModelPath):
            self.saver.restore(sess, tfModelPath)
            logging.info("Warmstart from %s", tfModelPath)
        #self.loadEmbeddingsTf(sess)
        self.initop = tf.initialize_all_variables()
        #sess.run(tf.initialize_variables(self.trainableVarList))
	sess.run(self.initop, feed_dict={self.place: self.embeds})
        saveGap = min(self.flags.batches/10, 100)
        for xiter in xrange(self.flags.batches):
            wid1, wid2, wid12, y = self.getRandomCompMiniBatch()
            _, trloss, trlossreg = sess.run([self.trainop, self.loss, self.lossreg],
                                            feed_dict={self.w1p: wid1,
                                                       self.w2p: wid2,
                                                       self.w12p: wid12,
                                                       self.yp: y
                                                       })
            if xiter % saveGap == 0:
                self.saver.save(sess, tfModelPath)
                teloss, telossreg = sess.run([self.loss, self.lossreg],
                                             feed_dict={self.w1p: self.caW1,
                                                        self.w2p: self.caW2,
                                                        self.w12p: self.caW12,
                                                        self.yp: self.caY
                                                        })
                logging.info("xiter=%d trloss=%g trlossreg=%g teloss=%g telossreg=%g",
                            xiter, trloss, trlossreg, teloss, telossreg)
        logging.info("Train end.")
        self.saver.save(sess, tfModelPath)
        self.saveModelNumpy(sess)

    def doEvalDimaNumpy(self, _w1, _w2, _w12, _y):
        """100% numpy eval diagnostics for Dima8."""
        logging.info('Begin %s', self.doEvalDimaNumpy.__name__)
        logging.info("w1 %d w2 %d w12 %d", len(_w1), len(_w2), len(_w12))
        npM1, npM2 = self.loadModelNumpy()
        _v1 = self.embeds[_w1]
        _v2 = self.embeds[_w2]
        _v12 = self.embeds[_w12]
        logging.info("v1 %s v2 %s v12 %s", str(_v1.shape), str(_v2.shape), str(_v12.shape))
        v12norms = np.linalg.norm(_v12, axis=1)
        preds = np.matmul(_v1, npM1) + np.matmul(_v2, npM2)
        predsNorm2 = np.linalg.norm(preds, axis=1)
        rowDots = np.sum(np.multiply(_v12, preds), axis=1)
        rowCosines = rowDots / v12norms / predsNorm2
        rowDiscrepancies = np.ones_like(rowCosines) - rowCosines
        rowLosses = np.abs(rowDiscrepancies - _y)
        avgLoss = np.average(rowLosses)
        logging.info('Average loss %g', avgLoss)
        plt.plot(_y, rowDiscrepancies, 'o')
        plt.show()
        xToYs = dict()
        for ix in xrange(len(_y)):
            xToYs[','.join((self.vocabArray[_w1[ix]].word, \
                self.vocabArray[_w2[ix]].word))] = [_y[ix], rowDiscrepancies[ix]]
        nccEval = EvalRecallPrecisionF1()
        res = nccEval.doEvalDict(xToYs)
        for perfKey in ['ktau', 'pearson', 'rho']:
            logging.info('%s %g', perfKey, res[perfKey])
        _, ax = plt.subplots()
        nccEval.plotRecallPrecision(ax, res['recall'], res['precision'], '', 'k')
        plt.show()
        logging.info('End %s', self.doEvalDimaNumpy.__name__)

    def doEval(self, sess, _w1, _w2, _w12, _y, doPlot=True):
        """Runs on test or all folds.  Saves to a dict with key = 'mod,head'
        and values = gold label, system label.  Optionally plot recall-precision."""
        logging.info('Begin %s', self.doEval.__name__)
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel.tf')
        assert os.path.exists(tfModelPath), 'Cannot load ' + tfModelPath
        self.saver.restore(sess, tfModelPath)
        logging.info("Loaded model from %s", tfModelPath)

        # Loads numpy embeddings into embedstf but also randomly
        # initializes the model.
        #sess.run(self.initop, feed_dict={self.place: self.embeds})
        # Only loads numpy embeddings into embedstf.
        sess.run(tf.initialize_variables([self.embedstf]), feed_dict={self.place: self.embeds})

        if self.flags.alltest:
            self.moveTrainToTestFold()

        avgLoss, rowLosses, rowDiscrepancies = \
            sess.run([self.loss,
                      self.rowLosses,
                      self.rowDiscrepancies],
                     feed_dict={self.w1p: _w1,
                                self.w2p: _w2,
                                self.w12p: _w12,
                                self.yp: _y,
                                })
        logging.info('avgLoss=%g', avgLoss)
        xToYs = dict()
        for ix in xrange(len(_y)):
            xToYs[','.join((self.vocabArray[_w1[ix]].word, \
                            self.vocabArray[_w2[ix]].word))] = \
                [_y[ix], rowDiscrepancies[ix]]
        outPicklePath = os.path.join(tfModelDirPath, 'predictions.pickle')
        with open(outPicklePath, 'wb') as outPickleFile:
            pickle.dump(xToYs, outPickleFile)
        logging.info('End %s', self.doEval.__name__)
        if not doPlot: return
        # first plot predicted vs observed discrepancy
        _, ax = plt.subplots()
        ax.scatter(_y, rowDiscrepancies)
        # then plot precision vs recall with decretized discrepancy
        nccEval = EvalRecallPrecisionF1()
        res = nccEval.doEvalDict(xToYs)
        for perfKey in ['ktau', 'pearson', 'rho']:
            logging.info('%s %g', perfKey, res[perfKey])
        for k, v in res.iteritems():
            if k != 'recall' and k != 'precision':
                logging.debug("result %s %g", k, v)
        _, ax = plt.subplots()
        nccEval.plotRecallPrecision(ax, res['recall'], res['precision'], '', 'k')
        plt.show()


def mainDima(flags):
    with tf.Session() as sess:
        nc = NccDiscrepancy(flags)
        if nc.flags.train: nc.doTrainDiscriminative(sess)
        logging.info('==== Assess on train fold ====')
        #nc.doEvalDimaNumpy(nc.trWid1, nc.trWid2, nc.trWid12, nc.trY)
        nc.doEval(sess, nc.trWid1, nc.trWid2, nc.trWid12, nc.trY)
        logging.info('==== Assess on test fold ====')
        #nc.doEvalDimaNumpy(nc.teWid1, nc.teWid2, nc.teWid12, nc.teY)
        nc.doEval(sess, nc.teWid1, nc.teWid2, nc.teWid12, nc.teY)
        fittedWeights = sess.run(nc.trainableVarList)
        print(fittedWeights)


def mainSigYaz(flags):
    with tf.Session() as sess:
        nc = NccDiscrepancy(flags)
        if nc.flags.train: nc.doTrainDiscriminative(sess)
        if nc.flags.alltest:
            nc.doEval(sess, nc.teWid1, nc.teWid2, nc.teWid12, nc.teY)
        else:
            logging.info('==== Assess on train fold ====')
            nc.doEval(sess, nc.trWid1, nc.trWid2, nc.trWid12, nc.trY)
            logging.info('==== Assess on test fold ====')
            nc.doEval(sess, nc.teWid1, nc.teWid2, nc.teWid12, nc.teY)


def mainYazdani(flags):
    with tf.Session() as sess:
        nc = NccDiscrepancy(flags)
        if nc.flags.train: nc.doTrain(sess)
        nc.doEval(sess, nc.caW1, nc.caW2, nc.caW12, nc.caY)


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True, type=str,
                        help='/path/to/vocab/and/embeddings/')
    parser.add_argument("--phrases", required=True, type=str,
                        help='/path/to/labeled/phrases.csv')
    parser.add_argument("--net", required=True, type=str,
                        help="salehi|dima|yazdani|sigyaz")
    parser.add_argument("--model", required=True, type=str,
                        help='/dir/holding/model/and/predictions')
    parser.add_argument("--train", nargs='?', const=True, default=False, type=bool,
                        help='Train using labeled instances?')
    parser.add_argument("--warm", nargs='?', const=True, default=False, type=bool,
                        help='Warm start from saved model?')
    parser.add_argument("--unsuper", nargs='?', const=True, default=False, type=bool,
                        help='Sample phrase alphabet for training?')
    parser.add_argument("--alltest", nargs='?', const=True, default=False, type=bool,
                        help='Apply model to all instances?')
    parser.add_argument('--batches', default=1000, type=int,
                        help='Number of training batches.')
    parser.add_argument("--unit", required=False, action='store_true',
                        help="Normalize embeddings to unit L2 length?")
    args = parser.parse_args()
    if args.net == 'salehi': mainDima(args)
    elif args.net == 'dima': mainDima(args)
    elif args.net == 'sigyaz' : mainSigYaz(args)
    elif args.net == 'yazdani': mainYazdani(args)
