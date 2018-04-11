# Implementation of dima classifier

#python ncc_cascade_tf_objective.py --glove /data/srijayd/local_data/glove_embeddings/ --phrases /data/srijayd/local_data/f_r1_r2_th/th/th.csv --model /data/srijayd/models/cascademodel/ --resultDir /data/srijayd/cascade/result/cascade/ --train True - Don't run this command for now


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

import tensorflow as tf

from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord, Phrase

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
        self.phrasesPath = flags.phrases
        self.phrasesDir = os.path.dirname(os.path.realpath(self.phrasesPath))
	self.loadThCompoundsCsv(self.phrasesPath)
	self.makeFoldsCompounds()
        logging.info('phrasesDir=%s', self.phrasesDir)
        # first load CSV and split train test
        self.buildLossCascade()
	print "Done"
        logging.info('End %s.__init__', NccDiscrepancy.__name__)

    def loadThCompoundsCsv(self, thCsvPath): # To load compounds from TH dataset for 'cascade' model	
	print "Loading TH compounds"
        self.allw1,self.allw2,self.allw12,self.ally = [], [], [], []
	self.numInst,self.numDatInst = 0,0
	classes = set([])
        with open(thCsvPath, mode='r') as th:
            for row in csv.reader(th, delimiter=','):
		assert len(row) == 5
		self.numInst += 1
		sPhrase = "_".join([row[0], row[1]]) #phrase in Glove is seperated by _ 
                iv1, iv2, iv3 = row[0] in self.termToTid, row[1] in self.termToTid, sPhrase in self.termToTid #vocabmap => termToTid for glove
		if not iv1 or not iv2 or not iv3: continue
		wMod, wHead, wPhrase = self.termToTid[row[0]], self.termToTid[row[1]], self.termToTid[sPhrase]
		self.allw1.append(wMod)
                self.allw2.append(wHead)
		self.allw12.append(wPhrase)
		if(row[3] == "OBJECTIVE"): #Making noncomp class
			row[3] = "NONCOMP"
                self.ally.append(row[3])
		classes.add(row[3])
		self.numDatInst += 1
	    classes = list(classes)
	    self.numClasses = len(classes)
	    noncomp_ind = classes.index("NONCOMP")
	    classes[noncomp_ind],classes[self.numClasses-1] = classes[self.numClasses-1],classes[noncomp_ind] #Swapping to put noncomp class to last
    	    classtoids = {}
	    i=0
	    for cs in classes:
	    	classtoids[cs] = i
		i += 1
	    for i in xrange(self.numDatInst):
	    	self.ally[i] = classtoids[self.ally[i]]
	    ids = list(xrange(self.numClasses))
 	    self.oneHot43 = tf.one_hot(ids,self.numClasses)
	    self.Classes = classes
	    assert classes[len(classes)-1] == "NONCOMP"
	    print "Found ",str(self.numDatInst)," from ",str(self.numInst)
    
    def makeFoldsCompounds(self): # To make train and test folds
	ind = list(xrange(self.numDatInst))
	self.deterministicShuffle(ind)
	train = 0.5 #As did by Dima
	dev = 0.3
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
	#self.clW1.extend(self.cdW1)
	#self.clW2.extend(self.cdW2)
	#self.clW12.extend(self.cdW12)
	#self.clY.extend(self.cdY)
	self.caW1, self.caW2, self.caW12, self.caY = [self.allw1[x] for x in ap], [self.allw2[x] for x in ap], [self.allw12[x] for x in ap], [self.ally[x] for x in ap]
	print "Folding Finished"

    def getRandomCompMiniBatch(self, batchSize=2000):
        sample = np.random.randint(0, len(self.clY), batchSize)
        sWid1 = [self.clW1[x] for x in sample]
        sWid2 = [self.clW2[x] for x in sample]
	sWid12 = [self.clW12[x] for x in sample]
        sY = [self.clY[x] for x in sample]
        return sWid1, sWid2, sWid12, sY

    def sigmoid_01(self,x):
	tw = tf.constant(1.0)
	tw = tf.sub(tw,1.0*tf.sigmoid(x-self.mean))
	return tw

    def buildLossCascade(self):
        self.w1 = tf.placeholder(tf.int32, [None]) #input embedding id 1 
        self.w2 = tf.placeholder(tf.int32, [None]) #input embedding id 2
	self.w12 = tf.placeholder(tf.int32, [None]) #input embedding id phrase12
        self.y = tf.placeholder(tf.int32, [None])#output class
        w1v = tf.nn.embedding_lookup(self.embedstf, self.w1) #Embedding table for words
        w2v = tf.nn.embedding_lookup(self.embedstf, self.w2) 
	w12v = tf.nn.embedding_lookup(self.embedstf, self.w12) 
        yh = tf.nn.embedding_lookup(self.oneHot43, self.y) #One hot encodings embedding table for classes

	self.numClasses -= 1 #To take first (n-1) classes as comp

	#N1
        W = tf.Variable(tf.random_normal(shape=[self.dim, self.numClasses],mean=0,stddev= 1. / math.sqrt(self.dim * (self.numClasses))))
	b = tf.Variable(tf.random_normal(shape=[self.numClasses],mean=0,stddev= 1. / math.sqrt(self.numClasses))) 
	Wcomp = tf.Variable(tf.random_normal(shape=[2*self.dim,self.dim],mean=0,stddev= 1. / math.sqrt(2 * self.dim * self.dim)))
	bcomp = tf.Variable(tf.random_normal(shape=[self.dim],mean=0,stddev= 1. / math.sqrt(self.dim)))
	predn1 = tf.matmul(tf.concat(1,[w1v,w2v]),Wcomp) + bcomp

	#N2 
        M1 = self.buildVariable([self.dim, self.dim], 1. / math.sqrt(self.dim * self.dim) , 'M1') #numDim->dim for glove
        M2 = self.buildVariable([self.dim, self.dim], 1. / math.sqrt(self.dim * self.dim) , 'M2')
        predPhrase = tf.matmul(w1v, M1) + tf.matmul(w2v, M2)
        diffs = tf.sub(predPhrase, w12v)
        diffs2 = tf.mul(diffs, diffs)
	compclasslosses = tf.cast(tf.reduce_sum(diffs2, 1),tf.float32)
	#self.meanlosses = tf.reduce_mean(compclasslosses)
	self.mean = tf.Variable(40.0)
	compclassscores = tf.map_fn(self.sigmoid_01,compclasslosses)

	#N1-N2 merged loss
	predcompclass = tf.transpose(tf.transpose(tf.nn.softmax(tf.matmul(predn1, W) + b))*compclassscores)
	make_noncomp = lambda x : 1-x
	noncompclassscores = tf.map_fn(make_noncomp,compclassscores) #making non-comp scores
	noncompclassscores = tf.transpose(tf.expand_dims(noncompclassscores,0)) #Adding one dimension and taking concatenation
	predcompclass = tf.concat(1,[predcompclass,noncompclassscores]) #appending non-comp class to end
	self.predcompclass = predcompclass #used for prediction
	
	yh = tf.cast(yh,tf.float32)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predcompclass,yh)
        cross_entropy = tf.reduce_mean(cross_entropy)
        self.loss = cross_entropy
        sgd = tf.train.AdagradOptimizer(.8)
        self.trainop = sgd.minimize(self.loss, var_list=[W,b,Wcomp,bcomp,M1,M2,self.mean])
        self.initop = tf.initialize_all_variables()
        self.saver = tf.train.Saver(tf.trainable_variables())
        logging.info("Built Cascaded network.")


    def doTrain(self, sess, maxIters=600):
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
	print "Model path is ",tfModelDirPath
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel_ob.tf')
        if os.path.exists(tfModelPath):
            self.saver.restore(sess, tfModelPath)
            logging.info("Train Warmstart from %s", tfModelPath)
	    print "Welcome to training"
        sess.run(self.initop, feed_dict={self.place: self.embeds})
	print "Training Started"
	for xiter in xrange(maxIters):
		wid1, wid2, wid12, y = self.getRandomCompMiniBatch()
		_, trloss = sess.run([self.trainop, self.loss],
                                            feed_dict={self.w1: wid1,
                                                       self.w2: wid2,
						       self.w12: wid12,
                                                       self.y: y
                                                       })
            	if xiter % 100 == 0:
                	self.saver.save(sess, tfModelPath)
                	teloss = sess.run(self.loss,
                                             	     feed_dict={self.w1: self.caW1,
                                                        	self.w2: self.caW2,
						       		self.w12: self.caW12,
                                                        	self.y: self.caY
                                                        	})
                	logging.info("CascadeModel Objective xiter=%d trloss=%g teloss=%g",xiter, trloss, teloss)
    
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
	
    def calClass(self,sess,w1,w2,w12,y):
	probs = sess.run(self.predcompclass,feed_dict={self.w1: [w1],self.w2: [w2],self.w12: [w12]})
	predClass = np.argmax(probs[0])
	return predClass

    def getConfusionMatrix(self,sess,w1s,w2s,w12s,ys):
	l = len(ys)
	correctlabels = 0
	predicted = []
	for i in xrange(l):
		predicted.append(self.calClass(sess,w1s[i],w2s[i],w12s[i],ys[i]))
	return confusion_matrix(ys,predicted)

    def getF1Scores(self,sess,w1s,w2s,w12s,ys):
	l = len(ys)
	correctlabels = 0
	predicted = []
	for i in xrange(l):
		predicted.append(self.calClass(sess,w1s[i],w2s[i],w12s[i],ys[i]))
	return f1_score(ys,predicted, average=None)

    def getClassesAccuracies(self,cm): 
	ind = 0
	classaccuracies = []
	for arr in cm:
		classaccuracies.append((cm[ind][ind]*1.0)/sum(cm[ind]))
		ind += 1
	return classaccuracies

    def getAccuracy(self,cm): #Input is confusion Matrix
	total = 0
	correct = 0
	ind = 0
	for arr in cm:
		correct += cm[ind][ind]
		total += sum(cm[ind])
		ind += 1
	return (correct*1.0)/total

    
    def makeResult(self,resultPath,cm,classF1Scores):
	outcmfile = open(resultPath+"result_cascade.csv",'w')
	outcmfile.write("Total Accuracy => ," + str(self.getAccuracy(cm)) + "\n\n")
	outcmfile.write("Class F1 Scores are => ")
	for ind in xrange(len(classF1Scores)):
		outcmfile.write(self.Classes[ind] + "," + str(classF1Scores[ind]) + "\n")
	outcmfile.write("\n")
	outcmfile.write("Confusion Matrix =>,")
	headClasses = ""
	for clss in self.Classes:
		headClasses = headClasses + clss + ","
	headClasses = headClasses[:len(headClasses)-1]
	outcmfile.write(headClasses + "\n")
	ind = 0
	for row in cm:
		line = self.Classes[ind] + ","
		for score in cm[ind]:
			line = line + str(score) + ","
		line = line[:len(line)-1]
		outcmfile.write(line + "\n")
		ind+=1
	outcmfile.close()
	outbarchart = resultPath + "barchart_cascade.png" #output png file
	y_pos = np.arange(len(list(self.Classes)))
	#classAccuracies = self.getClassesAccuracies(cm)
	assert len(y_pos) == len(classF1Scores)
	plt.bar(y_pos, classF1Scores,width=0.5,alpha=0.5)
	plt.xticks(y_pos,self.Classes,rotation='vertical')
	plt.ylabel('F1-Scores')
	plt.title('Class F1 Scores') 
	plt.show()
	plt.savefig(outbarchart)


    def doEval(self,sess):
    	print "Welcome to Cascade Objective Eval"
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel_ob.tf')
        assert os.path.exists(tfModelPath), 'Cannot load ' + tfModelPath
        self.saver.restore(sess, tfModelPath)
        logging.info("Warmstart from %s", tfModelPath)

        #INIT sess.run(tf.initialize_variables([self.embedstf]))
        #sess.run(self.initop, feed_dict={self.place: self.embeds})

	sess.run(tf.initialize_variables([self.embedstf]),feed_dict={self.place: self.embeds})     #For pretrained word embeddings

        if self.flags.alltest:
            self.moveTrainToTestFold()

	
	#Confusion Matrix to get Accuracies

	cm = self.getConfusionMatrix(sess,self.caW1,self.caW2,self.caW12,self.caY)
	f1scores = self.getF1Scores(sess,self.caW1,self.caW2,self.caW12,self.caY)

	print "Total Accuracy => ",str(self.getAccuracy(cm))

	print "Class Accuracies are "
	print self.getClassesAccuracies(cm)

	print "Class F1 Scores are "
	print f1scores
	
	self.makeResult(self.flags.resultDir,cm,f1scores)

	

def maincascade(flags):
    with tf.Session() as sess:
	print "Welcome to Cascade Objective classifier"
        nc = NccDiscrepancy(flags)
	print nc.flags.train
        if nc.flags.train: nc.doTrain(sess)
        nc.doEval(sess)


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
                        help='/path/to/cascade/model')
    parser.add_argument("--train", nargs='?', const=True, default=False, type=bool,
                        help='Train using labeled instances?')
    parser.add_argument("--alltest", nargs='?', const=True, default=False, type=bool,
                        help='Apply model to all instances?')
    parser.add_argument("--resultDir", required=True, type=str,
                        help='/path/to/cascade/model/result/') #default - /data/srijayd/cascade/result/cascade
    args = parser.parse_args()
    maincascade(args)







