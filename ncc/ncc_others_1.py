# Simple baselines inspired by Reddy and Salehi.

import os, sys, math, logging, argparse
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord, Phrase

#python ncc_others_1.py --indir /data/soumen/ncc/glove300/ --phrases /data/srijayd/local_data/f_r1_r2_th/r1/r1.csv --outdir /data/srijayd/local_data/f_r1_r2_th/r1

class NccOthers(NonComp):
    def __init__(self, flags):
        logging.info('Begin %s.__init__', NccOthers.__name__)
        super(NccOthers, self).__init__(flags)
        self.loadPhraseInstancesNcCsv(flags.phrases)
        logging.info('End %s.__init__', NccOthers.__name__)

    def testUnitNorm(self, narr):
        ones = np.ones_like(narr);
        print np.linalg.norm(ones-narr, ord=1)

    def runLinearSum(self, modMul, headMul):
        """
        Estimated phrase vector is convex combination of modifier
         and head vectors.  Should cover both comp1 and comp2
         cases of Salehi et al.
        :param modMul:
        :param headMul:
        :return:
        """
        vec1 = self.embeds[self.allWid1]
        norms1 = np.linalg.norm(vec1, axis=1)
        self.testUnitNorm(norms1)
        vec2 = self.embeds[self.allWid2]
        norms2 = np.linalg.norm(vec2, axis=1)
        self.testUnitNorm(norms2)
        vec12 = self.embeds[self.allWid12]
	vec12norm = np.linalg.norm(vec12, axis=1)
        norms12 = np.linalg.norm(vec12, axis=1)
        self.testUnitNorm(norms12)
        est12 = modMul * vec1 + headMul * vec2
	est12norm = np.linalg.norm(est12, axis=1)
        mul12 = np.multiply(est12, vec12)
        dot12 = np.sum(mul12, axis=1)
	vec12est12norm = vec12norm * est12norm
        cos12 = dot12 / vec12est12norm
        # Larger dot is judged more compositional.
        xToYs = dict()
        for ix in xrange(len(dot12)):
            xToYs[','.join((self.vocabArray[self.allWid1[ix]].word,  
                            self.vocabArray[self.allWid1[ix]].word))] = \
                [self.allY[ix], 1. - cos12[ix]] 
        outPicklePath = os.path.join(self.flags.outdir, 'linsum.pickle')
        with open(outPicklePath,'wb') as outPickleFile:
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
    args = parser.parse_args()
    nccOthers = NccOthers(args)
    nccOthers.runLinearSum(.5, .5)
