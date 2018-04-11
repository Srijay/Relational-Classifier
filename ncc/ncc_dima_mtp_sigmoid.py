# Command to run ncc_dat.py => python ncc_dima_mtp_sigmoid.py --indir /data/soumen/word2vec/phrasedot_dj/ --phrases /data/soumen/word2vec/phrasedot_dj/phrases_farahmand/farahmand_noncomp.csv --model /data/srijayd/ncc/src/ncc/phrases_farahmand/dima_fle/ --alltest True

import os, sys, math, logging, argparse
import numpy as np
import cPickle as pickle
import math
import matplotlib

matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, logistic, spearmanr
from enum import Enum

#import tensorflow as tf

from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord, Phrase

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

class NccInTestFold(Enum):
    """DATs are trained on compositional phrases only.
    Move this fraction of NCCs to the test fold."""
    nothing = 1
    half = 2
    all = 3

class NccDiscrepancy(NonComp):

    def __init__(self, flags, thresh=0.5):
        logging.info('Begin %s.__init__', NccDiscrepancy.__name__)
	super(NccDiscrepancy, self).__init__(flags)
        self.thresh = thresh
        self.phrasesPath = flags.phrases
        self.phrasesDir = os.path.dirname(os.path.realpath(self.phrasesPath))
        logging.info('Here phrasesDir=%s', self.phrasesDir)
	self.loadPhraseInstancesNcCsv(self.phrasesPath)
        self.selectCompPhrases()
	self.batchSize = 50

    def getRandomCompMiniBatch(self):
        """Like NonComp.getRandomMiniBatch, restricted to comp train fold."""
        sample = np.random.randint(0, len(self.clY), self.batchSize)
        sWid1 = [self.clW1[x] for x in sample]
        sWid2 = [self.clW2[x] for x in sample]
        sWid12 = [self.clW12[x] for x in sample]
        sY = [self.clY[x] for x in sample]
        return sWid1, sWid2, sWid12, sY	

    def doTrainNumpy(self, maxIters=1000):
        modelDirPath = os.path.join(self.phrasesDir,self.flags.model)
        modelPath = os.path.join(modelDirPath, 'compmodel.tf')
        if os.path.exists(modelPath):
            logging.info("Warmstart from %s", modelPath)
	    print "Training Started"
        self.M1 = np.random.rand(self.numDim,self.numDim) #Random Initialisation of Matrices M1 and M2
	self.M2 = np.random.rand(self.numDim,self.numDim)
	#n = 1/self.batchSize # Learning rate
	n = 0.01
        for xiter in xrange(maxIters): #temporarily no of iterations set to 10000
	    #print "Into the iterations"
            wid1, wid2, wid12, y = self.getRandomCompMiniBatch()
	    diffM1 = np.zeros((self.numDim,self.numDim)) #Error accumulator initialised to zero
	    diffM2 = np.zeros((self.numDim,self.numDim))
	    for i in xrange(len(wid1)):
	    	w1v = self.embeds[wid1[i]]
		w2v = self.embeds[wid2[i]]
		w12v = self.embeds[wid12[i]]
		loss = self.M1 * w1v + self.M2 * w2v - w12v
		loss = loss * loss
		loss = sigmoid(np.sum(loss))
		diffM1 = diffM1 + (loss - y[i])*loss*(1-loss)*(self.M1 * w1v * w1v.transpose() + self.M2 * w2v * w1v.transpose() - w12v * w1v.transpose()) #Accumulating the error w.r.t. M1
	        diffM2 = diffM2 + (loss - y[i])*loss*(1-loss)*(self.M2 * w2v * w2v.transpose() + self.M1 * w1v * w2v.transpose() - w12v * w2v.transpose())
	    self.M1 = self.M1 - n*diffM1
	    self.M2 = self.M2 - n*diffM2
	    #print M1[0] # M1 and M2 are learned
            #print "Iteration " + str(xiter) + " done"	     
		
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
    
    def doEvalNumpy(self):

	modelDirPath = os.path.join(self.phrasesDir,self.flags.model)
        modelPath = os.path.join(modelDirPath, 'compmodel.tf')
        assert os.path.exists(tfModelPath), 'Cannot load ' + tfModelPath

        if self.flags.alltest:
            self.moveTrainToTestFold()

        w1l, w2l, w12l, yl = self.caW1, self.caW2, self.caW12, self.caY
	rowLosses = []

	for i in xrange(len(w1l)):
		predPhrase = self.M1 * w1l[i] + self.M2 * w2l[i]
		loss = predPhrase - w12l
		loss = loss * loss
		rowLosses.append(np.sum(loss))
  	
	sysScores = [(2. * logistic.cdf(rl) - 1.) for rl in rowLosses]		

        xToYs = dict()
        for ix in xrange(len(_y)):
            xToYs[','.join((self.vocabArray[_w1[ix]].word, \
                            self.vocabArray[_w2[ix]].word))] = [_y[ix], sysScores[ix]]

        outPicklePath = os.path.join(modelDirPath, 'pred.pickle')
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
	nc = NccDiscrepancy(flags)
        nc.doTrainNumpy()
        nc.doEvalNumpy()

if __name__ == "__main__":
    reload(sys)
    print "Welcome to numpy dima"
    sys.setdefaultencoding('utf-8')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True, type=str,
                        help='/path/to/vocab/and/embeddings/') 
    parser.add_argument("--phrases", required=True, type=str,
                        help='/path/to/labeled/phrases.csv')
    parser.add_argument("--model", required=True, type=str,
                        help='/dir/holding/model/and/predictions')
    parser.add_argument("--alltest", nargs='?', const=True, default=False, type=bool,
                        help='Apply model to all instances?')
    args = parser.parse_args()
    mainDima(args)
