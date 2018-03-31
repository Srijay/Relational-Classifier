import os, sys, math, logging, argparse
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord, Phrase

class NccOthers(NonComp):
    def __init__(self, flags):
        logging.info('Begin %s.__init__', NccOthers.__name__)
        super(NccOthers, self).__init__(flags)
        self.loadPhraseInstancesNcCsv(flags.phrases)
        logging.info('End %s.__init__', NccOthers.__name__)

    def diagnostics(self, msg, wpv):
        norms1 = np.linalg.norm(wpv, axis=1)
        avgNorm1 = np.average(norms1)
        print msg, avgNorm1

    def runLinearSum(self, modMul, headMul):

        vec1 = self.embeds[self.allWid1]

        vec2 = self.embeds[self.allWid2]

        vec12 = self.embeds[self.allWid12]

        vec1norm = np.linalg.norm(vec1, axis=1)
        vec2norm = np.linalg.norm(vec2, axis=1)
        vec12norm = np.linalg.norm(vec12, axis=1)

	#cosine similarity between vec1 & vec12
	g1 = np.sum(np.multiply(vec1,vec12),axis=1)/(vec1norm*vec12norm)

	#cosine similarity between vec2 & vec12
	g2 = np.sum(np.multiply(vec2,vec12),axis=1)/(vec2norm*vec12norm)

	#losses = 0.2*(g1) + (g2) # e1 + e2 in diagram

	losses = []

	totalnums = g1.shape[0]

	for i in xrange(totalnums):
		
	

	# Larger loss thend to non-compositional
        xToYs = dict()
        for ix in xrange(len(vec12)):
            xToYs[','.join((self.vocabArray[self.allWid1[ix]].word,
                            self.vocabArray[self.allWid2[ix]].word))] = \
                [self.allY[ix], -losses[ix]]
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
    args = parser.parse_args()
    nccOthers = NccOthers(args)
    nccOthers.runLinearSum(.4, .6)
    nccOthers.runMockPerfect()
    nccOthers.runMockRandom()
