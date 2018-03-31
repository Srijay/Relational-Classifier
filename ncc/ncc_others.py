import os, sys, math, logging, argparse
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord, Phrase
import scipy.stats as stats

#python ncc_others.py --indir /data/soumen/ncc/glove300/ --phrases /data/srijayd/local_data/f_r1_r2_th/r1/r1.csv --outdir /data/srijayd/local_data/f_r1_r2_th/r1

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

    def computeCosine(self,v1,v2):
	v1norm = np.linalg.norm(v1)
	v2norm = np.linalg.norm(v2)
        mul12 = np.multiply(v1, v2)
        dot12 = np.sum(mul12)
        # Larger dot and cos is judged more compositional.
        vec12est12norm = v1norm * v2norm
        cos12 = dot12 / vec12est12norm
	return cos12

    def calCoef(self,v1,v2,v12):

	est = 0.7*v1 + 0.3*v2
	cosine = self.computeCosine(est,v12)

	
	est1 = 0.1*v1 + 0.9*v2
	cosine1 = self.computeCosine(est1,v12)
	if((cosine1) > (cosine)):
		est = est1
		cosine = cosine1


	est1 = 0.9*v1 + 0.1*v2
	cosine1 = self.computeCosine(est1,v12)
	if((cosine1) > (cosine)):
		est = est1
		cosine = cosine1

	
	arr = np.linspace(0, 1, 10)
	l = len(arr)
	for i in range(0,l):
		for j in range(0,l):
			if((arr[i] + arr[j])==1):
				est1 = arr[i]*v1 + arr[j]*v2
				cosine1 = self.computeCosine(est1,v12)
				if((cosine1) > (cosine)):
					est = est1
					cosine = cosine1
	
			
	return est #Best result yet
	#return (0.7*v1 + 0.3*v2)
	
	
    def makeHardAssignment(self,y):
    	if(y <= 0.4):
		return 0
	else:
		return 1

    def runLinearSum(self, modMul, headMul):
        """
        Estimated phrase vector is convex combination of modifier
         and head vectors.  Should cover both comp1 and comp2
         cases of Salehi et al.; also similar to Reddy.
        :param modMul:
        :param headMul:
        :return:
        """
        vec1 = self.embeds[self.allWid1]
        self.diagnostics('vec1', vec1)
	print type(vec1)
	print "here ",len(vec1)
        vec2 = self.embeds[self.allWid2]
        #self.diagnostics('vec2', vec2)
        vec12 = self.embeds[self.allWid12]
        #self.diagnostics('vec12', vec12)
        vec12norm = np.linalg.norm(vec12, axis=1)

        print "here ",vec1.shape
	totals = vec1.shape[0]

        est12 = []

	for i in xrange(totals):
		est12.append(self.calCoef(vec1[i],vec2[i],vec12[i]))

        est12norm = np.linalg.norm(est12, axis=1)
        mul12 = np.multiply(est12, vec12)
        dot12 = np.sum(mul12, axis=1)

        # Larger dot and cos is judged more compositional.
        vec12est12norm = vec12norm * est12norm
        cos12 = dot12 / vec12est12norm
        xToYs = dict()
	#outfile = open("compound_predicted_scores.csv","w")
        for ix in xrange(len(vec12)):
            xToYs[','.join((self.vocabArray[self.allWid1[ix]].word,
                            self.vocabArray[self.allWid2[ix]].word))] = \
                [self.allY[ix], (1. - cos12[ix])/2]
	    #outfile.write(self.Mod[ix]+" "+self.Head[ix]+","+str((1.0 - cos12[ix])/2)+","+str(self.allY[ix]) + "\n")
	    #print "Actual is ",self.allY[ix]
	    #print "Predicted is ",(1.0 - cos12[ix])/2
	    #print "compound is ",self.Mod[ix]," ",self.Head[ix]
	#outfile.close()
	x2 = [1.0]*len(cos12)
	x2 = np.subtract(x2,cos12)
	tau, p_value = stats.kendalltau(self.allY, x2)
	print "ktau is ",tau
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
