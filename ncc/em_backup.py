# Implementation of EM method for non-compositionality detection

#python em_backup.py --glove /data/srijayd/local_data/glove_embeddings/ --phrases /data/srijayd/local_data/f_r1_r2_th/th/th.csv --model /data/srijayd/models/emmodel/ --train True --alltest True - Don't run this command for now


import os, sys, math, logging, argparse, csv, re
import numpy as np
import cPickle as pickle

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, logistic, spearmanr
from enum import Enum

import tensorflow as tf

from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord, Phrase

#numInst=0

def sigmoid_01(x):
	tw = tf.constant(2.0)
	tw = tf.sub(tw,2.0*tf.sigmoid(x))
	return tw

class NccInTestFold(Enum):
    """DATs are trained on compositional phrases only.
    Move this fraction of NCCs to the test fold."""
    nothing = 1
    half = 2
    all = 3


class NccDiscrepancy(NonComp):

    regularizer = 0.0001

    def __init__(self, flags ,thresh=0.5):
        logging.info('Begin %s.__init__', NccDiscrepancy.__name__)
	super(NccDiscrepancy, self).__init__(flags)
	self.phrasesPath = flags.phrases
        self.thresh = thresh
	#self.numInst=0
        self.phrasesPath = flags.phrases
        self.phrasesDir = os.path.dirname(os.path.realpath(self.phrasesPath))
	self.loadThCompoundsCsv(self.phrasesPath)
	#self.makeFoldsCompounds()
        logging.info('phrasesDir=%s', self.phrasesDir)
        # first load CSV and split train test
        self.buildErrorEM()
	print "Done"
        logging.info('End %s.__init__', NccDiscrepancy.__name__)

    def loadThCompoundsCsv(self, thCsvPath): # To load compounds from TH dataset for 'cascade' model	
	print "Loading TH compounds"
        self.allw1,self.allw2,self.allw12,self.ally = [], [], [], []
	self.numInst,self.numDatInst = 0,0
	classes = set([])
        with open(thCsvPath, mode='r') as th:
            for row in csv.reader(th, delimiter=','):
		#if(self.numDatInst== 8): break
		assert len(row) == 5
		self.numInst += 1
		sPhrase = "_".join([row[0], row[1]]) #phrase in Glove is seperated by _ 
                iv1, iv2, iv3 = row[0] in self.termToTid, row[1] in self.termToTid, sPhrase in self.termToTid #vocabmap => termToTid for glove
		if not iv1 or not iv2 or not iv3: continue
		wMod, wHead, wPhrase = self.termToTid[row[0]], self.termToTid[row[1]], self.termToTid[sPhrase]
		self.allw1.append(wMod)
                self.allw2.append(wHead)
		self.allw12.append(wPhrase)
		if(row[3] == "LEXICALIZED"): #Making noncomp class
			row[3] = "NONCOMP"
                self.ally.append(row[3])
		classes.add(row[3])
		self.numDatInst += 1
	    classes = list(classes)
	    self.numClasses = len(classes)
	    #noncomp_ind = classes.index("NONCOMP")
	    #classes[noncomp_ind],classes[self.numClasses-1] = classes[self.numClasses-1],classes[noncomp_ind] #Swapping to put noncomp class to last
    	    self.classtoids = {}
	    i=0
	    for cs in classes:
	    	self.classtoids[cs] = i
		i += 1
	    for i in xrange(self.numDatInst):
	    	self.ally[i] = self.classtoids[self.ally[i]]
	    ids = list(xrange(self.numClasses))
 	    self.oneHot43 = tf.one_hot(ids,self.numClasses)
	    print "Found ",str(self.numDatInst)," from ",str(self.numInst)
    
    def makeFoldsCompounds(self): # To make train and test folds
	ind = list(xrange(self.numDatInst))
	self.deterministicShuffle(ind)
	le, ap = np.array_split(ind, 2) # Folded into two
	le, ap = le.tolist(), ap.tolist()
	self.clW1, self.clW2, self.clW12, self.clY = [self.allw1[x] for x in le], [self.allw2[x] for x in le], [self.allw12[x] for x in le], [self.ally[x] for x in le]
	self.caW1, self.caW2, self.caW12, self.caY = [self.allw1[x] for x in ap], [self.allw2[x] for x in ap], [self.allw12[x] for x in ap], [self.ally[x] for x in ap]
	print "Folding Finished"

    def getRandomCompMiniBatch(self, batchSize=500):
        sample = np.random.randint(0, len(self.clY), batchSize)
        sWid1 = [self.clW1[x] for x in sample]
        sWid2 = [self.clW2[x] for x in sample]
	sWid12 = [self.clW12[x] for x in sample]
        sY = [self.clY[x] for x in sample]
        return sWid1, sWid2, sWid12, sY

    def buildErrorEM(self):
        self.w1 = tf.placeholder(tf.int32, self.numDatInst) #input embedding id 1 #Currently assuming batch size is 50
        self.w2 = tf.placeholder(tf.int32, self.numDatInst) #input embedding id 2
	self.w12 = tf.placeholder(tf.int32, self.numDatInst) #input embedding id phrase12
        self.y = tf.placeholder(tf.int32, self.numDatInst)#output class
        w1v = tf.nn.embedding_lookup(self.embedstf, self.w1) #Embedding table for words
        w2v = tf.nn.embedding_lookup(self.embedstf, self.w2) 
	w12v = tf.nn.embedding_lookup(self.embedstf, self.w12) 
        yh = tf.nn.embedding_lookup(self.oneHot43, self.y) #One hot encodings embedding table for classes

	self.numClasses -= 1 #To take first (n-1) classes as comp

	''' #N1 
	matSize = self.dim * (self.numClasses) 
        stddev = 1. / math.sqrt(matSize)
        W = tf.Variable(tf.random_normal(shape=[self.dim, self.numClasses],mean=0,stddev=stddev))
	matSize = self.numClasses  
        stddev = 1. / math.sqrt(matSize)
	b = tf.Variable(tf.random_normal(shape=[self.numClasses],mean=0,stddev=stddev)) #Can be improved
	matSize = 2 * self.dim * self.dim  
        stddev = 1. / math.sqrt(matSize)
	Wcomp = tf.Variable(tf.random_normal(shape=[2*self.dim,self.dim],mean=0,stddev=stddev))
	matSize = self.dim 
        stddev = 1. / math.sqrt(matSize)
	bcomp = tf.Variable(tf.random_normal(shape=[self.dim],mean=0,stddev=stddev))
        pred1 = tf.concat(1,[w1v,w2v]) #Concatenate
	pred = tf.matmul(pred1,Wcomp) + bcomp
        predcompclass = tf.nn.softmax(tf.matmul(pred, W) + b) #will be used for evaluation
'''
	#N2 
	matSize = self.dim * self.dim  #numDim->dim for glove
        stddev = 1. / math.sqrt(matSize)
        M1 = self.buildVariable([self.dim, self.dim], stddev, 'M1') #numDim->dim for glove
        M2 = self.buildVariable([self.dim, self.dim], stddev, 'M2') #numDim->dim for glove
        predPhrase = tf.matmul(w1v, M1) + tf.matmul(w2v, M2)
        diffs = tf.sub(predPhrase, w12v)
        diffs2 = tf.mul(diffs, diffs)
        self.errors = tf.reduce_sum(diffs2, 1,keep_dims=True)  
	self.errors = tf.cast(self.errors,tf.float32)
	#print errors.eval()
	self.k=10
	val,idic=tf.nn.top_k(tf.transpose(self.errors), self.k, sorted=False)	#doubt,declare idx, self word?
	self.fvl,self.fid=tf.nn.top_k(tf.transpose(self.errors), self.k, sorted=True)
	idx,dummy=tf.nn.top_k(idic, self.k, sorted=True)
	idx=tf.reverse(idx, [False,True])
	#print idx.get_shape()
	#print idx.eval()
	#print val.get_shape()
	#print val.eval()
	self.label=tf.ones([self.numDatInst,1])           #doubt,valuates error also?
	#print self.label
	#self.shp= tf.to_int64((self.y.get_shape())[0])
	#idic=tf.transpose(idx)
	self.delta= tf.sparse_tensor_to_dense(tf.SparseTensor(tf.transpose(tf.to_int64(idx)),values=tf.ones(self.k),shape=[self.numDatInst]))
	#print self.delta.get_shape()
	#print self.delta
	self.label1=tf.sub(self.label, tf.expand_dims(self.delta,1))
	#print self.label1

        self.loss =tf.reduce_sum(tf.mul(self.label,self.errors)) 
        sgd = tf.train.AdagradOptimizer(.1)
        self.trainop = sgd.minimize(self.loss, var_list=[M1,M2])
	
        self.initop = tf.initialize_all_variables()                #doubt--which variable?
        self.saver = tf.train.Saver(tf.trainable_variables())
        logging.info("Built Cascaded network.")


    def doTrain(self, sess, maxIters=31):
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
	prev =tf.ones([self.numDatInst,1])  
	for xiter in xrange(maxIters):
		_, lbl,noncp,errs = sess.run([self.trainop, self.label1,self.fid,self.errors],
                                            feed_dict={self.w1: self.allw1,    
                                                       self.w2: self.allw2,
						       self.w12: self.allw12,
                                                       self.y: self.ally
                                                       })
            	if (0<xiter<5 or xiter % 5 == 0):
                	self.saver.save(sess, tfModelPath)
                	'''teloss = sess.run(self.loss,
                                             	     feed_dict={self.w1: self.caW1,
                                                        	self.w2: self.caW2,
						       		self.w12: self.caW12,
                                                        	self.y: self.caY
                                                        	})
'''                	
			#self.chang= sess.run(tf.reduce_sum(tf.cast(tf.logical_xor(tf.cast(prev,tf.bool),tf.cast(lbl,tf.bool)),tf.int32)))
			print 'iter=',xiter,', label changed= ',sess.run(tf.reduce_sum(tf.cast(tf.logical_xor(tf.cast(prev,tf.bool),tf.cast(lbl,tf.bool)),tf.int32))),'/', self.numDatInst
			#self.temp = sess.run(tf.transpose(lbl))
			#print 'errors= ',sess.run(tf.transpose(errs))
			top= sess.run(tf.squeeze(noncp))
			esr= sess.run(tf.squeeze(tf.transpose(errs)))
	#tope= sess.run(tf.squeeze(errs))
			print [(esr[x],self.tidToTerm[self.allw12[x]]) for x in top]
		prev = lbl
	#print [self.tidToTerm[self.allw12[x]] for x in list(xrange(self.numDatInst))]
    
    def moveTrainToTestFold(self):
        """Empties training fold into test fold."""
        logging.warn("Moving %d instances from train to test", len(self.clY))
        self.caW1.extend(self.clW1)
        self.caW2.extend(self.clW2)
        self.caW12.extend(self.clW12)
        self.caY.extend(self.clY)
        del self.clW1[:]
        del self.clW2[:]
        del self.clW12[:]
        del self.clY[:]
        logging.warn("Updated train=%d test=%d folds", len(self.clY), len(self.caY))
	
    def calClass(self,sess,w1,w2, w12,y):
	probs = sess.run(self.predcompclass,feed_dict={self.w1: [w1],self.w2: [w2],self.w12: [w12]})
	predClass = np.argmax(probs[0])
	if(predClass == y):
		return 1
	else:
		return 0

    def doEval(self,sess):
    	print "Welcome to Eval"
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel.tf')
        assert os.path.exists(tfModelPath), 'Cannot load ' + tfModelPath
        self.saver.restore(sess, tfModelPath)
        logging.info("Warmstart from %s", tfModelPath)

        #INIT sess.run(tf.initialize_variables([self.embedstf]))
        #sess.run(self.initop, feed_dict={self.place: self.embeds})

	sess.run(tf.initialize_variables([self.embedstf]),feed_dict={self.place: self.embeds})     #For pretrained word embeddings

        if self.flags.alltest:
            self.moveTrainToTestFold()
	
	w1s, w2s, w12s, ys = self.caW1, self.caW2, self.caW12, self.caY
	l = len(self.caY)
	correctlabels = 0
	for i in xrange(l):
		correctlabels += self.calClass(sess,w1s[i],w2s[i],w12s[i],ys[i])

	print "The Truely predicted labels are ",str(correctlabels)," from ",str(l)," instances"
	precision = (correctlabels*1.0)/l
	print "Precision = ",precision

	

def mainem(flags):
    with tf.Session() as sess:
	print "Welcome to Cascade classifier"
        nc = NccDiscrepancy(flags)
	print nc.flags.train
        if nc.flags.train: nc.doTrain(sess)
        #nc.doEval(sess)


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







