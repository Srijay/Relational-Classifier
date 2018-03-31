# Implementation of EM method for non-compositionality detection

#python em_sigmoid.py --indir /data/soumen/ncc/glove300 --phrases /data/srijayd/local_data/f_r1_r2_th/th/fh1.csv --model /data/srijayd/models/emmodel/ --train True

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

    def loadThCompoundsCsv(self, thCsvPath): # To load compounds from TH dataset for 'EM' model	
	print "Loading TH compounds"
        self.allw1,self.allw2,self.allw12,self.ally = [], [], [],[]
	self.numInst,self.numDatInst = 0,0
	with open(thCsvPath, mode='r') as th:
            for row in csv.reader(th, delimiter=','):
		#if(self.numDatInst== 20): break
		#assert len(row) == 5
		self.numInst += 1
		sPhrase = "^".join([row[0],row[1],'']) #phrase in Glove is seperated by _ 
                iv1, iv2, iv3 = row[0] in self.vocabMap, row[1] in self.vocabMap, sPhrase in self.vocabMap #vocabmap => termToTid for glove
		if not iv1 or not iv2 or not iv3: continue
		wMod, wHead, wPhrase = self.vocabMap[row[0]], self.vocabMap[row[1]], self.vocabMap[sPhrase]
		self.allw1.append(wMod)
                self.allw2.append(wHead)
		self.allw12.append(wPhrase)
		self.ally.append(int(row[2]))
		self.numDatInst += 1
	print "Found ",str(self.numDatInst)," from ",str(self.numInst)

    def buildExpect(self):
        self.w1e = tf.placeholder(tf.int32, self.numDatInst) #input embedding id 1 #whole traz dataset is used
        self.w2e = tf.placeholder(tf.int32, self.numDatInst) #input embedding id 2
	self.w12e = tf.placeholder(tf.int32, self.numDatInst) #input embedding id phrase12
	self.M1e = tf.placeholder(tf.float32, shape=(self.numDim, self.numDim))
	self.M2e = tf.placeholder(tf.float32, shape=(self.numDim, self.numDim))
        w1ve = tf.nn.embedding_lookup(self.embedstf, self.w1e) #Embedding table for words
        w2ve = tf.nn.embedding_lookup(self.embedstf, self.w2e) 
	w12ve = tf.nn.embedding_lookup(self.embedstf, self.w12e) 
        
	prede = tf.matmul(w1ve, self.M1e) + tf.matmul(w2ve, self.M2e)
        diffe = tf.sub(prede, w12ve)
        diffe2 = tf.mul(diffe, diffe)
        self.errore = tf.reduce_sum(diffe2, 1,keep_dims=True)  
	self.errore = tf.cast(self.errore,tf.float32)      #check
	self.k=200
	self.initope = tf.initialize_all_variables()                
        logging.info("Built Expect network.")
	'''
	self.x = tf.Variable(2295.0)   
	self.adjerre= tf.sub(self.errore,self.x)   #check on console
	self.adjsigerre=tf.map_fn(self.sigmoid_01,self.adjerre); 
	self.reduerre = tf.reduce_sum(self.adjsigerre) 
	self.minobj=tf.square(tf.sub(self.reduerre,tf.constant((self.numDatInst-self.k)*1.0)))
	sgde = tf.train.AdagradOptimizer(.9)
        self.trainope = sgde.minimize(self.minobj, var_list=[self.x])
	#self.final_label=(self.sig_errm*self.k)/self.normfac
'''


    def sigmoid_01(self,y):
	tw = tf.constant(1.0)
	tw = tf.sub(tw,1.0*tf.sigmoid(y))
	return tw

    def buildMax(self):
        self.w1m = tf.placeholder(tf.int32, self.numDatInst) #input embedding id 1 #whole traz dataset is used
        self.w2m = tf.placeholder(tf.int32, self.numDatInst) #input embedding id 2
	self.w12m = tf.placeholder(tf.int32, self.numDatInst) #input embedding id phrase12
	self.labelm=tf.placeholder(tf.float32, shape=(self.numDatInst,1))
        w1vm = tf.nn.embedding_lookup(self.embedstf, self.w1m) #Embedding table for words
        w2vm = tf.nn.embedding_lookup(self.embedstf, self.w2m) 
	w12vm = tf.nn.embedding_lookup(self.embedstf, self.w12m) 
        
	matSize = self.numDim * self.numDim  
        stddev = 1. / math.sqrt(matSize)
        self.M1m = self.buildVariable([self.numDim, self.numDim], stddev, 'M1m') 
        self.M2m = self.buildVariable([self.numDim, self.numDim], stddev, 'M2m') 
        predm = tf.matmul(w1vm, self.M1m) + tf.matmul(w2vm, self.M2m)
        diffm = tf.sub(predm, w12vm)
        diffm2 = tf.mul(diffm, diffm)
        self.errorm = tf.reduce_sum(diffm2, 1,keep_dims=True)  
	self.errorm = tf.cast(self.errorm,tf.float32)      #errors for all DataInst, it is (numDatInst x 1) in size

	self.lossm =tf.reduce_sum(tf.mul(self.labelm,self.errorm))  #the loss function = final label * error for that phrase
        sgd = tf.train.AdagradOptimizer(.1)
        self.trainopm = sgd.minimize(self.lossm, var_list=[self.M1m,self.M2m])
		
        self.initopm = tf.initialize_all_variables()                
        #self.saver = tf.train.Saver(tf.trainable_variables())
        logging.info("Built Max network.")

    def f(self,x):
	adjerr = self.errs - x
	vfun = np.vectorize(logistic.cdf)
	sigerr = vfun(adjerr)
	errall = np.sum(sigerr)
	opt = errall - ((self.numDatInst-self.k)*1.0)
	return opt	


    def doTrain(self, sess1, sess2, maxIters=15):
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
	print "Model path is ",tfModelDirPath
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel200.tf')
        if os.path.exists(tfModelPath):
            #self.saver.restore(sess, tfModelPath)
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
                                                       self.w2m: self.allw2,
						       self.w12m: self.allw12,
						       self.labelm: sess2.run(prev_labl)	
                                                       })
			#print "Here"
			#print dlabel
			#print sess1.run(self.M1m)
			#print sess1.run(self.M2m)
		#for eiter in xrange(10):			
		self.errs = sess2.run(self.errore,feed_dict={self.w1e: self.allw1,    
                                               self.w2e: self.allw2,
					       self.w12e: self.allw12,
					       self.M1e: sess1.run(self.M1m),
					       self.M2e: sess1.run(self.M2m) 	
                                               })
    	
        	#self.saver.save(sess, tfModelPath)
		print 'iter=',xiter#,'eiter',eiter #finding label changes using Xor of prev and cur lbl
		#idex_noncomp= sess2.run(tf.squeeze(noncpidx))   #list of idexs which are non-comp
		allerrs= sess2.run(tf.squeeze(tf.transpose(self.errs)))   #error for all data instances
		#print [(allerrs[x],self.tidToTerm[self.allw12[x]]) for x in idex_noncomp]  #print error and phrase having topk largest errors
		v,ind=tf.nn.top_k(tf.transpose(self.errs), self.numDatInst, sorted=True)
		rev_idx=tf.reverse(ind, [False,True]) 
		idex_nc= sess2.run(tf.squeeze(ind))
		idex_revnc= sess2.run(tf.squeeze(rev_idx))
		#print "Top45"
		#print [(allerrs[x],self.tidToTerm[self.allw12[x]]) for x in idex_nc[:self.k]]
		#print "Bottom245"
		#print [(allerrs[x],self.tidToTerm[self.allw12[x]]) for x in idex_revnc[:self.k]]
		#print [(allerrs[x],self.tidToTerm[self.allw12[x]]) for x in xrange(self.numDatInst)]  	
		#print sess2.run(tf.transpose(sig))
		#print sess2.run(tf.reduce_mean(allerrs))
		#print "Current labels "
		#print cur_labl
		#print [(allerrs[x],self.tidToTerm[self.allw12[x]],self.ally[x]) for x in idex_nc]
		print "All labels :"
		for i in range(33):
			sum1 = 0
			for j in range(30):
				sum1 += self.ally[idex_nc[30*i+j]]
			print 30*(i+1)," ",sum1
		#sum2=0
		#for j in range(8):
		#	sum2+=self.ally[idex_nc[80+j]]
		#print "8 ",sum2
			
		ranges = [10,4000,1]
		k = optimize.brute(self.f,(ranges,), finish=None)
		print "X is "	
		print k
		adjerre= tf.sub(self.errs,tf.cast(tf.constant(k*1.0),tf.float32))   #check on console
		prev_labl=tf.map_fn(self.sigmoid_01,adjerre); 
		vlab,indlab=tf.nn.top_k(tf.transpose(prev_labl), self.numDatInst, sorted=True)		
		print sess2.run(vlab)
		#print "Errors are "
		#print errs
		#print "Object "
		#print obj
   

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


