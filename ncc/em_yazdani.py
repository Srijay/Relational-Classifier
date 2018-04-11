# Implementation of EM method for non-compositionality detection

#python em_yazdani.py --indir /data/soumen/ncc/glove300 --phrases /data/srijayd/local_data/f_r1_r2_th/th/fh1.csv --model /data/srijayd/models/emmodel/ --train True 

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
	self.errore = tf.cast(self.errore,tf.float32)      #errors for all DataInst, it is (numDatInst x 1) in size

	k=200    #prior belief on number of non-comps
	self.value,self.index=tf.nn.top_k(tf.transpose(self.errore), k, sorted=True)    #index of topk most erroreous entries
	idx,dummy=tf.nn.top_k(self.index, k, sorted=True)    #sort indexes in decending order
	idx=tf.reverse(idx, [False,True])    #convert decending to ascending order, needed for sparse tensor to work
	self.label=tf.ones([self.numDatInst,1])        #set all labels to 1,done in every iteration   
	self.delta= tf.sparse_tensor_to_dense(tf.SparseTensor(tf.transpose(tf.to_int64(idx)),values=tf.ones(k),shape=[self.numDatInst]))    #generate a list in which only topk erroreous entry is 1 rest is zero
	self.final_label=tf.sub(self.label, tf.expand_dims(self.delta,1)) # substract delta from label to set all labels to 1 except the ones present in topk erroreneous list

	self.initope = tf.initialize_all_variables()                
        #self.saver = tf.train.Saver(tf.trainable_variables())
        logging.info("Built Expect network.")

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


    def doTrain(self, sess1, sess2, maxIters=15):
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
	print "Model path is ",tfModelDirPath
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel.tf')
        if os.path.exists(tfModelPath):
            #self.saver.restore(sess, tfModelPath)
            logging.info("Warmstart from %s", tfModelPath)
	    print "Welcome to training"
        sess1.run(self.initopm, feed_dict={self.place: self.embeds})
	sess2.run(self.initope, feed_dict={self.place: self.embeds})
	print "Training Started"
	prev_labl =tf.ones([self.numDatInst,1])  
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
			
		cur_labl,noncpidx,errs = sess2.run([self.final_label,self.index,self.errore],
                                            feed_dict={self.w1e: self.allw1,    
                                                       self.w2e: self.allw2,
						       self.w12e: self.allw12,
						       self.M1e: sess1.run(self.M1m),
						       self.M2e: sess1.run(self.M2m) 	
                                                       })
            	
        	#self.saver.save(sess, tfModelPath)
		print 'iter=',xiter,', label changed= ',sess2.run(tf.reduce_sum(tf.cast(tf.logical_xor(tf.cast(prev_labl,tf.bool),tf.cast(cur_labl,tf.bool)),tf.int32))),'/', self.numDatInst   #finding label changes using Xor of prev and cur lbl
		allerrs= sess2.run(tf.squeeze(tf.transpose(errs)))
		v,ind=tf.nn.top_k(tf.transpose(errs), self.numDatInst, sorted=True)
		idex_nc= sess2.run(tf.squeeze(ind))
		#print [(allerrs[x],self.tidToTerm[self.allw12[x]],self.ally[x]) for x in idex_nc]
		print "All labels :"
		for i in range(25):
			sum1 = 0
			for j in range(30):
				sum1 += self.ally[idex_nc[30*i+j]]
			print 30*(i+1)," ",sum1

		#idex_noncomp= sess2.run(tf.squeeze(noncpidx))   #list of idexs which are non-comp
		#allerrs= sess2.run(tf.squeeze(tf.transpose(errs)))   #error for all data instances
		#print [(allerrs[x],self.tidToTerm[self.allw12[x]]) for x in idex_noncomp]  #print error and phrase having topk largest errors
		#print [(allerrs[x],self.tidToTerm[self.allw12[x]]) for x in xrange(self.numDatInst)] 		
		#print sess2.run(tf.transpose(cur_labl))
		#print sess.run(tf.transpose(errs))
		prev_labl = tf.constant(cur_labl)


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


