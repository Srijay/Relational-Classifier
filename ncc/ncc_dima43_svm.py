# Implementation of dima classifier

#python ncc_dima43_svm.py --indir /data/soumen/ncc/glove300 --phrases /data/srijayd/local_data/f_r1_r2_th/th/th.csv - Don't run this command for now

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
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from numpy import linalg as LA

from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord, Phrase


class NccInTestFold(Enum):
    """DATs are trained on compositional phrases only.
    Move this fraction of NCCs to the test fold."""
    nothing = 1
    half = 2
    all = 3


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
		sPhrase = "^".join([row[0],row[1],''])
                iv1, iv2, iv3 = row[0] in self.vocabMap, row[1] in self.vocabMap, sPhrase in self.vocabMap #vocabmap => termToTid for glove

		if not iv1 or not iv2: continue
		wMod, wHead = self.vocabMap[row[0]], self.vocabMap[row[1]]
                #wPhrase = self.vocabMap[sPhrase] 
		self.allw1.append(wMod)
                self.allw2.append(wHead)
                self.allClasses.append(row[3])
		self.allw12.append(0) #Just to avoid w12
		classes.add(row[3])
		self.numDatInst += 1
	    classes = list(classes)
	    self.numClasses = len(classes)
	    print self.numClasses
    	    classtoids = {}
	    i=0
	    for cs in classes:
	    	classtoids[cs] = i
		i += 1
	    for i in xrange(self.numDatInst):
	    	self.ally.append(classtoids[self.allClasses[i]])
	    self.Classes = classes
	    print "Found ",str(self.numDatInst)," from ",str(self.numInst)
	

    def getNormDiffVec(self,x):
	diffvec1 = self.embeds[self.allw1[x]]
	diffvec2 = self.embeds[self.allw2[x]]
	return np.concatenate((diffvec1/1,diffvec2/1))
	#return np.concatenate((diffvec1/LA.norm(diffvec1),diffvec2/LA.norm(diffvec2)))
	
    
    def makeFoldsCompounds(self): # To make train and test folds
	ind = list(xrange(self.numDatInst))
	self.deterministicShuffle(ind)
	train = 0.8
	dev = 0
	test = 0.2
	splitindextrain = int(math.floor(train*self.numDatInst))
	splitindexdev = int(math.floor(dev*self.numDatInst))
	le = ind[:splitindextrain]
	de = ind[splitindextrain:splitindextrain + splitindexdev]
	ap = ind[splitindextrain + splitindexdev:]
	le, de, ap = list(le), list(de), list(ap)
	print "here length of train,dev and test are "
	print len(le)
	print len(ap)
	self.clW1, self.clW2, self.clW12, self.clY = [self.allw1[x] for x in le], [self.allw2[x] for x in le], [self.allw12[x] for x in le], [self.ally[x] for x in le]
	self.cdW1, self.cdW2, self.cdW12, self.cdY = [self.allw1[x] for x in de], [self.allw2[x] for x in de], [self.allw12[x] for x in de], [self.ally[x] for x in de]
	self.caW1, self.caW2, self.caW12, self.caY = [self.allw1[x] for x in ap], [self.allw2[x] for x in ap], [self.allw12[x] for x in ap], [self.ally[x] for x in ap]
	self.X_train = [self.getNormDiffVec(x) for x in le]
	self.X_dev = [self.getNormDiffVec(x) for x in de]
	self.X_test = [self.getNormDiffVec(x) for x in ap]
	#self.X_train = [(self.embeds[self.allw1[x]] - self.embeds[self.allw2[x]]) for x in le]
	#self.X_test = [(self.embeds[self.allw1[x]] - self.embeds[self.allw2[x]]) for x in ap]
	print "Folding Finished"

    def checkSVMAccuracy(self,C1,gamma1):
	clft = svm.SVC(C=C1,gamma=gamma1)
	clft.fit(self.X_train,self.clY)
	pred_classes_dev = clft.predict(self.X_dev)
	return self.getAccuracy(confusion_matrix(self.cdY,list(pred_classes_dev)))

    def buildSvmModel(self):

	#svc = svm.SVC(C=1)
	self.clf = svm.LinearSVC()

        Cs = np.logspace(-6, -1, 10)

	tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.logspace(-9, 3, 13),
                     'C': np.logspace(-9, 3, 13)}]
        
        #self.clf = GridSearchCV(estimator=svc,param_grid=tuned_parameters,n_jobs=-1)
        
	self.clf.fit(self.X_train,self.clY)
	
	pred_classes_train = self.clf.predict(self.X_train)
	pred_classes_test = self.clf.predict(self.X_test)
	return confusion_matrix(self.clY,list(pred_classes_train)),confusion_matrix(self.caY,list(pred_classes_test))


    def getClassesAccuracies(self,cm): 
       	ind = 0
       	classaccuracies = []
       	for arr in cm:
         	classaccuracies.append(self.Classes[ind] + "=>" + str((cm[ind][ind]*1.0)/sum(cm[ind])))
         	ind += 1
      	return classaccuracies

    def getAccuracy(self,cm): 
	total = 0
	correct = 0
	ind = 0
	for arr in cm:
		correct += cm[ind][ind]
		total += sum(cm[ind])
		ind += 1
	return (correct*1.0)/total


    def runSVM(self):
    	print "Welcome to Eval"

	#Confusion Matrix to get Accuracies

	cm_train,cm_test = self.buildSvmModel()

	print "training accuracy is ",self.getAccuracy(cm_train)

	print "test accuracy is ",self.getAccuracy(cm_test)
		

def mainDima43(flags):
    print "Welcome to dima SVM classifier"
    nc = NccDiscrepancy(flags)
    nc.runSVM()

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


