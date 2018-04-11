

#python ncc_tratz_diagnostic.py --glove /data/srijayd/local_data/glove_embeddings/ --phrases /data/srijayd/local_data/f_r1_r2_th/th/th.csv

import os, sys, math, logging, argparse, csv, re
import numpy as np
import cPickle as pickle
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, logistic, spearmanr
from enum import Enum
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
from operator import add
from sklearn import preprocessing
from numpy import linalg as LA
from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord, Phrase


class NccDiscrepancy(NonComp):

    def __init__(self, flags ,thresh=0.5):
        logging.info('Begin %s.__init__', NccDiscrepancy.__name__)
	super(NccDiscrepancy, self).__init__(flags)
	self.phrasesPath = flags.phrases
        self.thresh = thresh
        self.phrasesPath = flags.phrases
        self.phrasesDir = os.path.dirname(os.path.realpath(self.phrasesPath))
	self.loadThCompoundsCsv(self.phrasesPath)
        logging.info('phrasesDir=%s', self.phrasesDir)
	print "Done"
        logging.info('End %s.__init__', NccDiscrepancy.__name__)


    def loadThCompoundsCsv(self, thCsvPath): # To load compounds from TH dataset	
	print "Loading TH compounds"
        self.allw1,self.allw2,self.allw12,self.allClasses,self.ally = [], [], [], [], []
	self.numInst,self.numDatInst = 0,0
	classes = set([])
        with open(thCsvPath, mode='r') as th:
            for row in csv.reader(th, delimiter=','):
	        assert len(row) == 5
		self.numInst += 1
		sPhrase = "_".join([row[0], row[1]])
                iv1, iv2, iv3 = row[0] in self.termToTid, row[1] in self.termToTid, sPhrase in self.termToTid #vocabmap => termToTid for glove
		if not iv1 or not iv2 or not iv3: continue
		wMod, wHead, wPhrase = self.termToTid[row[0]], self.termToTid[row[1]], self.termToTid[sPhrase] 
		self.allw1.append(wMod)
                self.allw2.append(wHead)
                self.allClasses.append(row[3])
		self.allw12.append(wPhrase)
		classes.add(row[3])
		self.numDatInst += 1
		#if(self.numDatInst==50):
		#	break
	    classes = list(classes)
	    self.numClasses = len(classes)
	    print self.numClasses
    	    classtoids = {}
	    self.classcnt={}	
	    self.classvecsum ={}	
	    i=0
	    for cs in classes:
	    	classtoids[cs] = i
		self.classcnt[i]=0
		self.classvecsum[i]=[[0.0]*300]
		i += 1
	    for i in xrange(self.numDatInst):
		classnum = classtoids[self.allClasses[i]]
	    	self.ally.append(classnum)
		self.classcnt[classnum]+=1
		self.getNormDiffVec(i)
		#print self.diffvec1
		self.classvecsum[classnum] = map(add,self.classvecsum[classnum],self.diffvec1)
		#print self.classvecsum[classnum] 
	    self.Classes = classes
	    print "Found ",str(self.numDatInst)," from ",str(self.numInst)
	

    def getNormDiffVec(self,x):
	diffve1 = self.embeds[self.allw12[x]]-self.embeds[self.allw2[x]]
	self.diffvec1 = preprocessing.normalize(diffve1, norm='l2')
	#diffvec2 = self.embeds[self.allw12[x]]-self.embeds[self.allw2[x]]
	#print self.diffvec1
	#return diffvec1
	
	
    def similar(self):
    	print "Welcome to solve part"
	res=[[0.0 for x in range(40)] for y in range(40)]
	for i in xrange(self.numClasses):
		rest=0
		for j in xrange(self.numClasses):
			x=self.classvecsum[i][0]
			#print x
			y=self.classvecsum[j][0]
			#print y
			#res[i][j]=0
			for k in range(300):
				res[i][j]+=(x[k]*y[k])  
			#print res[i][j]
		  	if(i==j):
			 #print self.classcnt[i]
			 res[i][j]-= self.classcnt[i]
			 if self.classcnt[i]>1:
			   res[i][j] /=(self.classcnt[i]*(self.classcnt[i]-1))
			 print res[i][j],' ',
		  	else:
			 res[i][j] /=(self.classcnt[i]*self.classcnt[j])
			 rest+= res[i][j]		
		  	#print res[i][j],' ',
		print rest/(self.numClasses-1)
	 
		

def mainDima43(flags):
    print "Welcome to dima SVM classifier"
    nc = NccDiscrepancy(flags)
    nc.similar()

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", required=True, type=str,
                        help='/path/to/vocab/and/embeddings/') 
    parser.add_argument("--phrases", required=True, type=str,
                        help='/path/to/labeled/phrases.csv')
    args = parser.parse_args() 
    mainDima43(args)


