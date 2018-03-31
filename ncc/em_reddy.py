# Implementation of EM method on reddy data

#python em_reddy.py --indir /mnt/infoback/data/soumen/ncc/glove300 --phrases /data/srijayd/local_data/f_r1_r2_th/r1_r2_literals.csv --model /data/srijayd/models/emmodel/ --train True --alltest True

import os, sys, math, logging, argparse, csv, re
import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib
from scipy import optimize


matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, logistic, spearmanr
from enum import Enum

from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord, Phrase


class NccDiscrepancy(NonComp):

    def __init__(self, flags):
        logging.info('Begin %s.__init__', NccDiscrepancy.__name__)
	super(NccDiscrepancy, self).__init__(flags)
	self.phrasesPath = flags.phrases
        self.phrasesDir = os.path.dirname(os.path.realpath(self.phrasesPath))
	self.loadThCompoundsCsv(self.phrasesPath)
        logging.info('phrasesDir=%s', self.phrasesDir)
	self.buildExpect()
        self.buildMax()
	print "Done"
        logging.info('End %s.__init__', NccDiscrepancy.__name__)

    def loadThCompoundsCsv(self, thCsvPath): # To make list of w1,w2,w12 id's and list of comp-noncomp judgement (non-comp=1 score)
	print "Loading FH compounds"
        self.allw1,self.allw12,self.ally,self.allylit = [], [], [],[]
	self.numInst,self.numDatInst = 0,0
	with open(thCsvPath, mode='r') as th:
            for row in csv.reader(th, delimiter=','):
		assert len(row) == 5
		self.numInst += 1
		sPhrase = "^".join([row[0], row[1], ''])
                iv1, iv2, iv3 = row[0] in self.vocabMap, row[1] in self.vocabMap, sPhrase in self.vocabMap 
		if not iv1 or not iv2 or not iv3: continue
		wMod, wHead, wPhrase = self.vocabMap[row[0]], self.vocabMap[row[1]], self.vocabMap[sPhrase]
		self.allw1.append(wMod)
		self.allw1.append(wHead)    #adding modifier and head to same list
		self.allw12.append(wPhrase)
		self.allw12.append(wPhrase) #adding phrase twice, one for head and other for modifier
		self.ally.append(row[2])  #append the non-comp/comp score also
		self.ally.append(row[2])
		self.allylit.append(row[3]) #appending literality score of modifier and head
		self.allylit.append(row[4])
		self.numDatInst += 2   #2 words are added by using one phrase
	print "Found ",str(self.numDatInst)," from ",str(self.numInst)

    def buildExpect(self):
        self.w1e = tf.placeholder(tf.int32, self.numDatInst) #input embedding id 1 #whole traz dataset is used
        self.w12e = tf.placeholder(tf.int32, self.numDatInst) #input embedding id phrase12
	self.M1e = tf.placeholder(tf.float32, shape=(self.numDim, self.numDim))   #matrix to pre-multiply the head/modifier vector
	self.M2e = tf.placeholder(tf.float32, shape=(self.numDim, self.numDim)) 	
	w1ve = tf.nn.embedding_lookup(self.embedstf, self.w1e) #Embedding table for words 
	w12ve = tf.nn.embedding_lookup(self.embedstf, self.w12e) 
        
	prede1 = tf.matmul(w1ve, self.M1e)
	prede2 = tf.matmul(w12ve, self.M2e)		
	diffcon=tf.sub(prede1,prede2)    # (w1*M1-w12)
	diffconsq=tf.mul(diffcon,diffcon)

        self.errore = tf.reduce_sum(diffconsq, 1,keep_dims=True)  
	self.errore = tf.cast(self.errore,tf.float32)      
	self.k=180  # half of compounds are non-compositional
	self.initope = tf.initialize_all_variables()                
        logging.info("Built Expect network.")

    def sigmoid_01(self,y):   #label needed for loss function , here comp score ==1 , loss function try to fit it
	one = tf.constant(1.0)
	negation = tf.sub(one,1.0*tf.sigmoid(y))
	return negation
	
    def buildMax(self): #finding labels
        self.w1m = tf.placeholder(tf.int32, self.numDatInst) #input embedding id 1 #whole traz dataset is used
	self.w12m = tf.placeholder(tf.int32, self.numDatInst) #input embedding id phrase12
	self.labelm=tf.placeholder(tf.float32, shape=(self.numDatInst,1))
        w1vm = tf.nn.embedding_lookup(self.embedstf, self.w1m) #Embedding table for words 
	w12vm = tf.nn.embedding_lookup(self.embedstf, self.w12m) 
        
	matSize = self.numDim * self.numDim 
        stddev = 1. / math.sqrt(matSize)
        self.M1m = self.buildVariable([self.numDim, self.numDim], stddev, 'M1m')
	self.M2m = self.buildVariable([self.numDim, self.numDim], stddev, 'M2m')
        predm1 = tf.matmul(w1vm, self.M1m)
	predm2 = tf.matmul(w12vm, self.M2m)
        diffm = tf.sub(predm1, predm2)
	diffm2 = tf.mul(diffm, diffm)
        self.errorm = tf.reduce_sum(diffm2, 1,keep_dims=True)  
	self.errorm = tf.cast(self.errorm,tf.float32)      #errors for all DataInst, it is (numDatInst x 1) in size

	self.lossm =tf.reduce_sum(tf.mul(self.labelm,self.errorm))  #the loss function = final label * error for that phrase
        sgd = tf.train.AdagradOptimizer(.1)
        self.trainopm = sgd.minimize(self.lossm, var_list=[self.M1m,self.M2m])
		
        self.initopm = tf.initialize_all_variables() 
        logging.info("Built Max network.")
	

    def f(self,x): #Given errors, return summation of adjusted sigmoid for a variable of adjustment 'x', that we calculate by root finding
	adjerr = self.errsnumpy - x
	vecfun = np.vectorize(logistic.cdf)
	sigerr = vecfun(adjerr)
	errsum = np.sum(sigerr)
	opt = errsum - ((self.numDatInst-self.k)*1.0)
	return opt	


    def sigmoid_02(self,y):   #label needed for loss function , here comp score ==1 , loss function try to fit it
	negation = 1.0*tf.sigmoid(y)
	return negation



    def doTrain(self, sess1, sess2, maxIters=15):
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
	print "Model path is ",tfModelDirPath
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel.tf')
        if os.path.exists(tfModelPath):
            logging.info("Warmstart from %s", tfModelPath)
	    print "Welcome to training"
        sess1.run(self.initopm, feed_dict={self.place: self.embeds})
	sess2.run(self.initope, feed_dict={self.place: self.embeds})
	print "Training Started"
	prev_labl =tf.random_uniform([self.numDatInst,1])

	for xiter in xrange(maxIters):
		
		for miter in xrange(2):		
			_,dlabel = sess1.run([self.trainopm,tf.transpose(self.labelm)],
                                            feed_dict={self.w1m: self.allw1,
						       self.w12m: self.allw12,
						       self.labelm: sess2.run(prev_labl)	
                                                       })  
			
		self.errs = sess2.run(self.errore,feed_dict={self.w1e: self.allw1, 
					       self.w12e: self.allw12,
					       self.M1e: sess1.run(self.M1m),
					       self.M2e: sess1.run(self.M2m) 
                                               })
    		
        	
		print 'iter=',xiter
				
		allerrs= sess2.run(tf.squeeze(tf.transpose(self.errs)))   #error for all data instances, in list format
		v,ind=tf.nn.top_k(tf.transpose(self.errs), self.numDatInst, sorted=True) #sorted error, ind has sorted indexs of error 
		idex_nc= sess2.run(tf.squeeze(ind))  #indexs in list format

		for i in idex_nc:
			print self.allylit[i],' ',self.vocabArray[self.allw1[i]].word,' ',self.vocabArray[self.allw12[i]].word
		emin=tf.reduce_min(allerrs)   #for scaling error
		emax=tf.reduce_max(allerrs)	
		self.errs= tf.expand_dims((2.* allerrs - tf.ones_like(allerrs)*(emin+emax))*40.0/(emax-emin),1)
		self.errsnumpy = sess2.run(self.errs)
		ranges = [-100.,100.,.1]  #range to perform brute force search of root of shifted sigmoid
		k = optimize.brute(self.f,(ranges,), finish=None)  #actual root finding
		#k = bisect(self.f,100.0,5000.0)
		print "X is "	
		print k
		adjerre= tf.sub(self.errs,tf.cast(tf.constant(k*1.0),tf.float32))   
		prev_labl=tf.map_fn(self.sigmoid_01,adjerre)  #final label in this step, sigmoid of adjusted errors
 

def mainem(flags):
	sess1 = tf.Session()
	sess2 = tf.Session() 
	print "Welcome to EM classifier"
	nc = NccDiscrepancy(flags)
	print nc.flags.train
	if nc.flags.train: nc.doTrain(sess1,sess2)
	sess1.close()
	sess2.close()


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True, type=str,
                        help='/path/to/vocab/and/embeddings/') 
    parser.add_argument("--phrases", required=True, type=str,
                        help='/path/to/labeled/phrases.csv')
    parser.add_argument("--model", required=True, type=str,
                        help='/path/to/em/model')
    parser.add_argument("--train", nargs='?', const=True, default=False, type=bool,
                        help='Train using labeled instances?')
    parser.add_argument("--alltest", nargs='?', const=True, default=False, type=bool,
                        help='Apply model to all instances?')
    args = parser.parse_args()
    mainem(args)


