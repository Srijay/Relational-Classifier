
#python ncc_knearest_dima43.py --indir /data/soumen/ncc/glove300 --phrases /data/srijayd/local_data/f_r1_r2_th/th/th.csv

import os, sys, math, logging, argparse, csv, re
import numpy as np
import cPickle as pickle
import matplotlib
from numpy import linalg as LA

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
	self.makeFoldsCompounds()
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
		sPhrase = "^".join([row[0], row[1], ''])
                iv1, iv2, iv3 = row[0] in self.vocabMap, row[1] in self.vocabMap, sPhrase in self.vocabMap #vocabmap => termToTid for glove
		if not iv1 or not iv2 or not iv3: continue
		wMod, wHead, wPhrase = self.vocabMap[row[0]], self.vocabMap[row[1]], self.vocabMap[sPhrase] 
		self.allw1.append(wMod)
                self.allw2.append(wHead)
                self.allClasses.append(row[3])
		self.allw12.append(wPhrase)
		classes.add(row[3])
		self.numDatInst += 1
	    classes = list(classes)
	    self.numClasses = len(classes)
	    print self.numClasses
    	    classtoids = {}
	    self.classcnt=[0]*self.numClasses	
	    i=0
	    for cs in classes:
	    	classtoids[cs] = i
		i += 1
	    for i in xrange(self.numDatInst):
		classnum = classtoids[self.allClasses[i]]
	    	self.ally.append(classnum)
		self.classcnt[classnum]+=1
	    self.Classes = classes
	    print "Found ",str(self.numDatInst)," from ",str(self.numInst)

    def makeFoldsCompounds(self): # To make train and test folds
	ind = list(xrange(self.numDatInst))
	self.deterministicShuffle(ind)
	train = 0.8
	test = 0.2
	splitindextrain = int(math.floor(train*self.numDatInst))
	self.le = ind[:splitindextrain]
	self.ap = ind[splitindextrain:]
	self.le,self.ap = list(self.le),list(self.ap)
	self.caY = [self.ally[x] for x in self.ap]
	print "here length of train,dev and test are "
	print len(self.le)
	print len(self.ap)
	print "Folding Finished"

    def getSimilarity(self,trainIndex,testIndex):
	train1 = self.embeds[self.allw1[trainIndex]]
	train2 = self.embeds[self.allw2[trainIndex]]
	train1_norm = LA.norm(train1)
	train2_norm = LA.norm(train2)
	test1 = self.embeds[self.allw1[testIndex]]
	test2 = self.embeds[self.allw2[testIndex]]
	test1_norm = LA.norm(test1)
	test2_norm = LA.norm(test2)
	return np.dot(train1/train1_norm,test1/test1_norm) + np.dot(train2/train2_norm,test2/test2_norm)
	
    def evalNearestNeighbours(self):
    	print "Welcome to solve part"
	classTrainCount = [0]*len(self.le)
	
	for trainInstance in self.le:
		classTrainCount[self.ally[trainInstance]]+=1

	self.predicted = []
	for testInstance in self.ap:
		self.classSimil = [0.0]*self.numClasses
		for trainInstance in self.le:
			classIndex = self.ally[trainInstance]
			self.classSimil[classIndex] += self.getSimilarity(trainInstance,testInstance)
		for i in range(0,self.numClasses):
			self.classSimil[i] = (self.classSimil[i]*1.0)/classTrainCount[i]
		self.predicted.append(np.argmax(self.classSimil))
	return confusion_matrix(self.caY,self.predicted)
	
    def runNearestNeighbour(self):
	print "Running Model"
	cm_test = self.evalNearestNeighbours()
	print "test accuracy is ",self.getAccuracy(cm_test)

    def getAccuracy(self,cm): 
	total = 0
	correct = 0
	ind = 0
	for arr in cm:
		correct += cm[ind][ind]
		total += sum(cm[ind])
		ind += 1
	return (correct*1.0)/total


def mainDima43(flags):
    print "Welcome to dima Nearest Neighbour classifier"
    nc = NccDiscrepancy(flags)
    nc.runNearestNeighbour()


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True, type=str,
                        help='/path/to/vocab/and/embeddings/') 
    parser.add_argument("--phrases", required=True, type=str,
                        help='/path/to/labeled/phrases.csv')
    args = parser.parse_args() 
    mainDima43(args)


