# Implementation of EM method for non-compositionality detection

#python em_final.py --glove /data/srijayd/local_data/glove_embeddings/ --phrases /data/srijayd/local_data/f_r1_r2_th/th/em_test.csv --model /data/srijayd/models/emmodel/ --train True --alltest True 

import os, sys, math, logging, argparse, csv, re
import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib

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
        self.buildErrorEM()
	print "Done"
        logging.info('End %s.__init__', NccDiscrepancy.__name__)

    def loadThCompoundsCsv(self, thCsvPath): # To load compounds from TH dataset for 'EM' model	
	print "Loading TH compounds"
        self.allw1,self.allw2,self.allw12 = [], [], []
	self.numInst,self.numDatInst = 0,0
	with open(thCsvPath, mode='r') as th:
            for row in csv.reader(th, delimiter=','):
		if(self.numDatInst== 20): break
		assert len(row) == 5
		self.numInst += 1
		sPhrase = "_".join([row[0], row[1]]) #phrase in Glove is seperated by _ 
                iv1, iv2, iv3 = row[0] in self.termToTid, row[1] in self.termToTid, sPhrase in self.termToTid #vocabmap => termToTid for glove
		if not iv1 or not iv2 or not iv3: continue
		wMod, wHead, wPhrase = self.termToTid[row[0]], self.termToTid[row[1]], self.termToTid[sPhrase]
		self.allw1.append(wMod)
                self.allw2.append(wHead)
		self.allw12.append(wPhrase)
		self.numDatInst += 1
	print "Found ",str(self.numDatInst)," from ",str(self.numInst)

    def buildErrorEM(self):
        self.w1 = tf.placeholder(tf.int32, self.numDatInst) #input embedding id 1 #whole traz dataset is used
        self.w2 = tf.placeholder(tf.int32, self.numDatInst) #input embedding id 2
	self.w12 = tf.placeholder(tf.int32, self.numDatInst) #input embedding id phrase12
        w1v = tf.nn.embedding_lookup(self.embedstf, self.w1) #Embedding table for words
        w2v = tf.nn.embedding_lookup(self.embedstf, self.w2) 
	w12v = tf.nn.embedding_lookup(self.embedstf, self.w12) 
        
	#Error finding
	matSize = self.dim * self.dim  #numDim->dim for glove
        stddev = 1. / math.sqrt(matSize)
        self.M1 = self.buildVariable([self.dim, self.dim], stddev, 'M1') 
        self.M2 = self.buildVariable([self.dim, self.dim], stddev, 'M2') 
        predPhrase = tf.matmul(w1v, self.M1) + tf.matmul(w2v, self.M2)
        diffs = tf.sub(predPhrase, w12v)
        diffs2 = tf.mul(diffs, diffs)
        self.errors = tf.reduce_sum(diffs2, 1,keep_dims=True)  
	self.errors = tf.cast(self.errors,tf.float32)      #errors for all DataInst, it is (numDatInst x 1) in size

	#E-step
	k=6   #prior belief on number of non-comps
	self.value,self.index=tf.nn.top_k(tf.transpose(self.errors), k, sorted=True)    #index of topk most erroreous entries
	idx,dummy=tf.nn.top_k(self.index, k, sorted=True)    #sort indexes in decending order
	idx=tf.reverse(idx, [False,True])    #convert decending to ascending order, needed for sparse tensor to work
	self.label=tf.ones([self.numDatInst,1])        #set all labels to 1,done in every iteration   
	self.delta= tf.sparse_tensor_to_dense(tf.SparseTensor(tf.transpose(tf.to_int64(idx)),values=tf.ones(k),shape=[self.numDatInst]))    #generate a list in which only topk erroreous entry is 1 rest is zero
	self.final_label=tf.sub(self.label, tf.expand_dims(self.delta,1)) # substract delta from label to set all labels to 1 except the ones present in topk erroreneous list

	#M-step
        self.loss =tf.reduce_sum(tf.mul(self.final_label,self.errors))  #the loss function = final label * error for that phrase
        sgd = tf.train.AdagradOptimizer(.1)
        self.trainop = sgd.minimize(self.loss, var_list=[self.M1,self.M2])
	#self.trainop = self.fun(sgd,M1,M2)
	
        self.initop = tf.initialize_all_variables()                
        self.saver = tf.train.Saver(tf.trainable_variables())
        logging.info("Built EM network.")


    def doTrain(self, sess, maxIters=5):
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
	print "Model path is ",tfModelDirPath
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel.tf')
        if os.path.exists(tfModelPath):
            self.saver.restore(sess, tfModelPath)
            logging.info("Warmstart from %s", tfModelPath)
	    print "Welcome to training"
        sess.run(self.initop, feed_dict={self.place: self.embeds})
	print "Training Started"
	prev_labl =tf.ones([self.numDatInst,1])  
	for xiter in xrange(maxIters):
		_, cur_labl,noncpidx,errs = sess.run([self.trainop, self.final_label,self.index,self.errors],
                                            feed_dict={self.w1: self.allw1,    
                                                       self.w2: self.allw2,
						       self.w12: self.allw12
                                                       })
            	if (0<xiter<5 or xiter % 1 == 0):
                	self.saver.save(sess, tfModelPath)
			print 'iter=',xiter,', label changed= ',sess.run(tf.reduce_sum(tf.cast(tf.logical_xor(tf.cast(prev_labl,tf.bool),tf.cast(cur_labl,tf.bool)),tf.int32))),'/', self.numDatInst   #finding label changes using Xor of prev and cur lbl
			idex_noncomp= sess.run(tf.squeeze(noncpidx))   #list of idexs which are non-comp
			allerrs= sess.run(tf.squeeze(tf.transpose(errs)))   #error for all data instances
			#print [(allerrs[x],self.tidToTerm[self.allw12[x]]) for x in idex_noncomp]  #print error and phrase having topk largest errors
			print [(allerrs[x],self.tidToTerm[self.allw12[x]]) for x in xrange(self.numDatInst)] 
			print sess.run(tf.transpose(cur_labl))
			#print sess.run(tf.transpose(errs))
			#print sess.run(self.M1)
			#print sess.run(self.M2)
		prev_labl = cur_labl
   

def mainem(flags):
    with tf.Session() as sess:
	print "Welcome to EM classifier"
        nc = NccDiscrepancy(flags)
	print nc.flags.train
        if nc.flags.train: nc.doTrain(sess)


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", required=True, type=str,
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


