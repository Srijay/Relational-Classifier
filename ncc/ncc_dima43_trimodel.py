# Implementation of dima classifier

#python ncc_dima43_trimodel.py --indir /mnt/c/Users/srdeshp/Desktop/MLApps/Relational-Classifier/Embeddings/Wikipedia_Gigaword/glove.6B.300d.txt --phrases /mnt/c/Users/srdeshp/Desktop/MLApps/Relational-Classifier/f_r1_r2_th/th/th.csv --model /mnt/c/Users/srdeshp/Desktop/MLApps/Relational-Classifier/model/trineural_model_simul/ --resultDir /mnt/c/Users/srdeshp/Desktop/MLApps/Relational-Classifier/ --train true

#train - 70% dev - 10% test - 20%

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
	self.model_name = 'glove_wikipedia_300.tf'
	self.loadThCompoundsCsv(self.phrasesPath)
	self.makeFoldsCompounds()
	self.encodeEmbeddings()
	print("Encodings are made")
        logging.info('phrasesDir=%s', self.phrasesDir)
        # first load CSV and split train test
	self.embedstf = tf.Variable(self.place, name='embedstf', trainable=False)
        self.buildLossDima43()
	print "Done"
        logging.info('End %s.__init__', NccDiscrepancy.__name__)

    def loadThCompoundsCsv(self, thCsvPath): # To load compounds from TH dataset	
	print "Loading TH compounds"
        self.allw1,self.allw2,self.allw12,self.ally = [], [], [], []
	self.numInst,self.numDatInst = 0,0
	classes = set([])
        with open(thCsvPath, mode='r') as th:
            for row in csv.reader(th,delimiter=','):
		assert len(row) == 5
		self.numInst += 1
		sPhrase = "^".join([row[0],row[1],''])
                iv1, iv2, iv3 = row[0] in self.vocabMap, row[1] in self.vocabMap, sPhrase in self.vocabMap #vocabmap => termToTid for glove
		if not iv1 or not iv2: continue
		wMod, wHead = self.vocabMap[row[0]], self.vocabMap[row[1]] 
		self.allw1.append(wMod)
                self.allw2.append(wHead)
                self.ally.append(row[3])
		self.allw12.append(0) #commented to ease code, change it to wPhrase when needed
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
	    	self.ally[i] = classtoids[self.ally[i]]
	    ids = list(xrange(self.numClasses))
 	    self.oneHot43 = tf.one_hot(ids,self.numClasses)
	    self.Classes = classes
	    print self.Classes
	    print classtoids
	    print "Found ",str(self.numDatInst)," from ",str(self.numInst)

    
    def makeFoldsCompounds(self): # To make train and test folds : Improved Sampling
	ind = list(xrange(self.numDatInst))
	self.deterministicShuffle(ind)
	train = 0.6 #As did by Dima
	dev = 0.2
	test = 0.2
	class_to_ids = {}

	for i in xrange(self.numDatInst):
		cls = self.ally[i]
		if cls in class_to_ids:
			class_to_ids[cls].append(i)
		else:
			class_to_ids[cls] = [i]

	le = []
	de = []
	ap = []

	for cls in class_to_ids:
		ind = class_to_ids[cls]
		self.deterministicShuffle(ind)
		splitindextrain = int(math.floor(train*len(ind)))
		splitindexdev = int(math.floor(dev*len(ind)))
		le=le+ind[:splitindextrain]
		de=de+ind[splitindextrain:splitindextrain + splitindexdev]
		ap=ap+ind[splitindextrain + splitindexdev:]

	print "here length of train,dev and test are "
	print len(le)
	print len(de)
	print len(ap)
	self.trainIds = le
	self.devIds = de
	self.testIds = ap 
	self.clW1, self.clW2, self.clW12, self.clY = [self.allw1[x] for x in le], [self.allw2[x] for x in le], [self.allw12[x] for x in le], [self.ally[x] for x in le]
	self.cdW1, self.cdW2, self.cdW12, self.cdY = [self.allw1[x] for x in de], [self.allw2[x] for x in de], [self.allw12[x] for x in de], [self.ally[x] for x in de]
	self.caW1, self.caW2, self.caW12, self.caY = [self.allw1[x] for x in ap], [self.allw2[x] for x in ap], [self.allw12[x] for x in ap], [self.ally[x] for x in ap]
	print "Folding Finished"

    def generateSamplesToAutoEncode(self,points,batchSize):
	sample = np.random.randint(0,len(points),batchSize)
	retdata = [points[x] for x in sample]
	return retdata

    def encodeEmbeddings(self):
	batchsize = 30
	encodeSize = self.numClasses
	maxIter = 200000
	modelpath = "/mnt/c/Users/srdeshp/Desktop/MLApps/Relational-Classifier/Embeddings/Encoded"

	embeddingsToEncode = []

	for i in self.trainIds:
		embeddingsToEncode.append(np.concatenate((self.embeds[self.allw1[i]],self.embeds[self.allw2[i]])))

	inputSize = len(embeddingsToEncode[0])
	encodings = self.autoEncode(embeddingsToEncode,batchsize,inputSize,encodeSize,maxIter,modelpath)
	i=0
	self.autoencodingsDict = {}
	for r in self.trainIds:
		self.autoencodingsDict[r] = encodings[i]
 
    def autoEncode(self, points, batchSize, inputSize, encodeSize, maxIter, modelpath):

	self.inputEmb = tf.placeholder(tf.float32, shape=[None,inputSize])

	hiddenlayerSize = 100

	W_in = tf.Variable(tf.random_normal(shape=[inputSize,hiddenlayerSize], mean=0.3, stddev=0.1))
	b_in = tf.Variable(tf.random_normal(shape=[hiddenlayerSize], mean=0, stddev=0.1))
	W_hidden_in = tf.Variable(tf.random_normal(shape=[hiddenlayerSize,encodeSize], mean=0.3, stddev=0.1))
	b_hidden_in = tf.Variable(tf.random_normal(shape=[encodeSize], mean=0, stddev=0.1))
	W_hidden_out = tf.Variable(tf.random_normal(shape=[encodeSize,hiddenlayerSize], mean=0.3, stddev=0.1))
	b_hidden_out = tf.Variable(tf.random_normal(shape=[hiddenlayerSize], mean=0, stddev=0.1))
	W_out = tf.Variable(tf.random_normal(shape=[hiddenlayerSize,inputSize], mean=0.3, stddev=0.1))
	b_out = tf.Variable(tf.random_normal(shape=[inputSize], mean=0, stddev=0.1))

	#neural constuction
	tfencoded_hidden_in = tf.map_fn(tf.nn.relu,tf.matmul(self.inputEmb,W_in) + b_in)
	self.tf_encoded = tf.matmul(tfencoded_hidden_in,W_hidden_in) + b_hidden_in
	tfencoded_hidden_out = tf.map_fn(tf.nn.relu,tf.matmul(self.tf_encoded,W_hidden_out) + b_hidden_out)
	reconstructed = tf.matmul(tfencoded_hidden_out,W_out) + b_out

	recon_loss = tf.reduce_sum(pow((self.inputEmb-reconstructed),2))

	trainop_encode = tf.train.AdamOptimizer(.01).minimize(recon_loss)
	initop_encode = tf.initialize_all_variables()
	saver_encode = tf.train.Saver()
	sess_encode = tf.Session()
	
	
	if(os.path.exists(modelpath+".index")):
		print("Restoring the autoencode model")
		saver_encode.restore(sess_encode,modelpath)
		print("Autoencode Model restored.")
	else:
		
		print("Training Starts")
		sess_encode.run(initop_encode)
		for i in xrange(maxIter):
			traindata = self.generateSamplesToAutoEncode(points,batchSize)    
			_,trainloss,trinput,trreconstructed = sess_encode.run([trainop_encode,recon_loss,W_in,W_out],feed_dict={inputEmb:traindata})
			if(i%1000==0):
				print(trainloss) 
		
		save_path = saver_encode.save(sess_encode,modelpath) 
		print("AutoEncode Model saved in path: %s" % save_path)
	
	encodes = sess_encode.run(self.tf_encoded,feed_dict={self.inputEmb:points})
	
	return encodes
    

    def buildLossDima43(self):
        self.w1 = tf.placeholder(tf.int32, [None]) #input embedding id 1
        self.w2 = tf.placeholder(tf.int32, [None]) #input embedding id 2
	self.w12 = tf.placeholder(tf.int32, [None]) #input embedding phrase 1_2
        self.y = tf.placeholder(tf.int32, [None])  #output class
        w1v = tf.nn.embedding_lookup(self.embedstf, self.w1) #Embedding table for words
        w2v = tf.nn.embedding_lookup(self.embedstf, self.w2) 
	#w12v = tf.nn.embedding_lookup(self.embedstf, self.w12) 
	self.w12v_encode = tf.placeholder(tf.float32, shape=(None, self.numClasses))

        yh = tf.nn.embedding_lookup(self.oneHot43, self.y) #One hot encodings embedding table for 43 classes
	yh = tf.cast(yh,tf.float32)

	#Neural Network 1
	self.allhidden = self.numDim
	self.hidden1 = 2*self.numDim

	matSize = self.hidden1 * self.numClasses  
        stddev = 1. / math.sqrt(matSize)
        W1 = tf.Variable(tf.random_normal(shape=[self.hidden1, self.numClasses],mean=0,stddev=stddev))

	matSize = self.numClasses  
        stddev = 1. / math.sqrt(matSize)
	b1 = tf.Variable(tf.random_normal(shape=[self.numClasses],mean=0,stddev=stddev))

	matSize = self.numDim * self.hidden1  
        stddev = 1. / math.sqrt(matSize)
	Wcomp1 = tf.Variable(tf.random_normal(shape=[self.numDim,self.hidden1],mean=0,stddev=stddev))

	matSize = self.hidden1 
        stddev = 1. / math.sqrt(matSize)
	bcomp1 = tf.Variable(tf.random_normal(shape=[self.hidden1],mean=0,stddev=stddev))

	#pred1_ = tf.concat(1,[w1v-w12v,w2v-w12v]) #Concatenate
	pred1_ = tf.matmul(w1v,Wcomp1) + bcomp1
	pred1_ = tf.map_fn(tf.nn.relu,pred1_)
	pred1 = tf.matmul(pred1_, W1) + b1
	#pred1 = tf.nn.softmax(pred1)

	#Neural Network 2

	self.hidden2 = 2*self.numDim

	matSize = self.hidden2 * self.numClasses  
        stddev = 1. / math.sqrt(matSize)
        W2 = tf.Variable(tf.random_normal(shape=[self.hidden2, self.numClasses],mean=0,stddev=stddev))

	matSize = self.numClasses  
        stddev = 1. / math.sqrt(matSize)
	b2 = tf.Variable(tf.random_normal(shape=[self.numClasses],mean=0,stddev=stddev))

	matSize = self.numDim * self.hidden2  
        stddev = 1. / math.sqrt(matSize)
	Wcomp2 = tf.Variable(tf.random_normal(shape=[self.numDim,self.hidden2],mean=0,stddev=stddev))

	matSize = self.hidden2 
        stddev = 1. / math.sqrt(matSize)
	bcomp2 = tf.Variable(tf.random_normal(shape=[self.hidden2],mean=0,stddev=stddev))

	#pred2_ = tf.concat(1,[(w1v-w12v)/w1v_w12v_norm,(w2v-w12v)/w2v_w12v_norm]) #Concatenate
	pred2_ = tf.matmul(w2v,Wcomp2) + bcomp2
	pred2_ = tf.map_fn(tf.nn.relu,pred2_)
	pred2 = tf.matmul(pred2_, W2) + b2
	#pred2 = tf.nn.softmax(pred2)



	#Neural Network 3

	self.hidden3 = 2*self.numDim

	matSize = self.hidden3 * self.numClasses  
        stddev = 1. / math.sqrt(matSize)
        W3 = tf.Variable(tf.random_normal(shape=[self.hidden3, self.numClasses],mean=0,stddev=stddev))

	matSize = self.numClasses  
        stddev = 1. / math.sqrt(matSize)
	b3 = tf.Variable(tf.random_normal(shape=[self.numClasses],mean=0,stddev=stddev))

	matSize = self.numClasses * self.hidden3  
        stddev = 1. / math.sqrt(matSize)
	Wcomp3 = tf.Variable(tf.random_normal(shape=[self.numClasses,self.hidden3],mean=0,stddev=stddev))

	matSize = self.hidden3 
        stddev = 1. / math.sqrt(matSize)
	bcomp3 = tf.Variable(tf.random_normal(shape=[self.hidden3],mean=0,stddev=stddev))

        #pred3 = tf.concat([w1v,w2v],1) #Concatenate
	pred3 = self.w12v_encode
	pred3_ = tf.matmul(pred3,Wcomp3) + bcomp3
	pred3_ = tf.map_fn(tf.nn.relu,pred3_)
	pred3 = tf.matmul(pred3_, W3) + b3
	#pred3 = tf.nn.softmax(pred3)

	#self.alpha1 = tf.Variable(tf.random_normal(shape=[1,self.numClasses],mean=0,stddev=1./math.sqrt(self.numClasses)))
	#self.alpha2 = tf.Variable(tf.random_normal(shape=[1,self.numClasses],mean=0,stddev=1./math.sqrt(self.numClasses)))
	#self.alpha3 = tf.Variable(tf.random_normal(shape=[1,self.numClasses],mean=0,stddev=1./math.sqrt(self.numClasses)))

	#pred = self.alpha1*pred1 + self.alpha2*pred2 + self.alpha3*pred3
	pred = pred3
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels = yh)
        cross_entropy = tf.reduce_mean(cross_entropy)
        self.loss = cross_entropy
        sgd = tf.train.AdamOptimizer(0.1)
	#sgd = tf.train.FtrlOptimizer(.2,l1_regularization_strength=0.8,l2_regularization_strength=0.5)
        #self.trainop = sgd.minimize(self.loss, var_list=[self.alpha1,self.alpha2,self.alpha3,Wcomp1,bcomp1,W1,b1,Wcomp2,bcomp2,W2,b2,Wcomp3,bcomp3,W3,b3])
	self.trainop = sgd.minimize(self.loss, var_list=[Wcomp3,bcomp3,W3,b3])

	self.predprob = pred #will be used for prediction
	
        self.initop = tf.initialize_all_variables()
        self.saver = tf.train.Saver(tf.trainable_variables())
        logging.info("Built Dima43 network.")

    def getRandomCompMiniBatch(self, batchSize=28):
        sample = np.random.randint(0, len(self.clY), batchSize)
        sWid1 = [self.clW1[x] for x in sample]
        sWid2 = [self.clW2[x] for x in sample]
	sWid12 = [self.clW12[x] for x in sample]
        sY = [self.clY[x] for x in sample]
	sTrainIds = [self.trainIds[x] for x in sample]
        return sWid1, sWid2, sWid12, sY, sTrainIds

    def doTrain(self, sess, maxIters=100000):
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
	print "Model path is ",tfModelDirPath
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
	sess.run(self.initop, feed_dict={self.place: self.embeds})

        tfModelPath = os.path.join(tfModelDirPath,self.model_name)

        if (0):
        	self.saver.restore(sess, tfModelPath)
            	logging.info("Warmstart from %s", tfModelPath)
	    	print "Welcome to training"

	    	print "Training Started"
		prevAccuracy = self.get_3nn_accuracy(sess,self.cdW1,self.cdW2,self.cdW12,self.cdY)
	    	
	    	print "previous accuracy is ",prevAccuracy
	else:
	    	prevAccuracy = 0

	count = 0
	for xiter in xrange(maxIters):
		wid1, wid2, wid12, y, trainSampleIds = self.getRandomCompMiniBatch()
		trainEncodings = [self.autoencodingsDict[i] for i in trainSampleIds]
		_, trloss = sess.run([self.trainop, self.loss],
                                            feed_dict={self.w1: wid1,
                                                       self.w2: wid2,
						       self.w12: wid12,
                                                       self.y: y,
						       self.w12v_encode: trainEncodings
                                                       })
            	if xiter % 100 == 0:

			

			if(1):

				embeddingsToEncode = []

				for i in self.testIds:
					embeddingsToEncode.append(np.concatenate((self.embeds[self.allw1[i]],self.embeds[self.allw2[i]])))
				
				testEncodings = sess.run(self.tf_encoded,feed_dict={self.inputEmb:embeddingsToEncode})

				teloss = sess.run(self.loss,feed_dict={self.w1: self.caW1,
									self.w2: self.caW2,
									self.w12: self.caW12,
									self.y: self.caY,
						       			self.w12v_encode: testEncodings
									})
				
				#Code for Early stopping for Dima43
				logging.info("Dima43 xiter=%d trloss=%g teloss=%g",xiter, trloss,teloss) #trloss and teloss are numpy ndarrays of one element
			
				currAccuracy = self.get_3nn_f1(sess,self.cdW1,self.cdW2,self.cdW12,self.cdY,self.devIds)

				if(currAccuracy > prevAccuracy):
					logging.info("")
					prevAccuracy = currAccuracy
					logging.info("Validation F1 => %g",prevAccuracy)
					testAccuracy = self.get_3nn_f1(sess,self.caW1,self.caW2,self.caW12,self.caY,self.testIds)
					logging.info("Test F1 => %g",testAccuracy)
					self.saver.save(sess, tfModelPath)	
					logging.info("")
					count = 0
				else:
					count += 1
					if(count > 200):
						break
				#tAccuracy = self.get_3nn_accuracy(sess,self.clW1,self.clW2,self.clW12,self.clY)
				#print "training accuracy is ",tAccuracy
				#if(tAccuracy > 0.92):
					#break


    def moveTrainToTestFold(self):
        """Empties training fold into test fold."""
        logging.warn("Moving %d instances from train to test", len(self.clY))
        self.caW1.extend(self.clW1)
        self.caW2.extend(self.clW2)
	self.caW12.extend(self.clW12)
        self.caY.extend(self.clY)
        del self.clW1[:]
        del self.clW2[:]
        logging.warn("Updated train=%d test=%d folds", len(self.clY), len(self.caY))

    def calClass(self,sess,w1,w2,w12,y,rowid):
	encoding = sess.run(self.tf_encoded,feed_dict={self.inputEmb:[np.concatenate((self.embeds[self.allw1[rowid]],self.embeds[self.allw2[rowid]]))]})
	probs = sess.run(self.predprob,feed_dict={self.w1: [w1],self.w2: [w2], self.w12v_encode: encoding})
	predClass = np.argmax(probs[0])
	return predClass

    def get_3nn_f1(self,sess,w1s,w2s,w12s,ys,rowids):
	l = len(ys)
	correctlabels = 0
	predicted = []
	for i in xrange(l):
		predicted.append(self.calClass(sess,w1s[i],w2s[i],w12s[i],ys[i],rowids[i]))
	return f1_score(ys, predicted, average='micro')

    def getConfusionMatrix(self,sess,w1s,w2s,w12s,ys):
	l = len(ys)
	correctlabels = 0
	predicted = []
	for i in xrange(l):
		predicted.append(self.calClass(sess,w1s[i],w2s[i],w12s[i],ys[i]))
	return confusion_matrix(ys,predicted)

    def getMicroAveragedF1Score(self,sess,w1s,w2s,w12s,ys):
	l = len(ys)
	correctlabels = 0
	predicted = []
	for i in xrange(l):
		predicted.append(self.calClass(sess,w1s[i],w2s[i],w12s[i],ys[i]))

	return f1_score(ys, predicted, average='micro')

    def getClassesAccuracies(self,cm): 
       	ind = 0
       	classaccuracies = []
	ofile = open("class_accuracy_0_0_1.csv","w")
       	for arr in cm:
         	classaccuracies.append(self.Classes[ind]+"=>"+str((cm[ind][ind]*1.0)/sum(cm[ind])))
		towrite = self.Classes[ind],",",str((cm[ind][ind]*1.0)/sum(cm[ind]))
		ofile.write(str(towrite))
		ofile.write("\n")
         	ind += 1
	ofile.close()
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


    def doEval(self,sess):

    	print "Welcome to Eval"
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath,self.model_name)
        #assert os.path.exists(tfModelPath), 'Cannot load ' + tfModelPath
        self.saver.restore(sess, tfModelPath)
        logging.info("Warmstart from %s", tfModelPath)

        #sess.run(self.initop)
	a1 = sess.run(self.alpha1)
	a2 = sess.run(self.alpha2)
	a3 = sess.run(self.alpha3)

	sess.run(tf.initialize_variables([self.embedstf]),feed_dict={self.place: self.embeds})     #For pretrained word embeddings

        #if self.flags.alltest:
            #self.moveTrainToTestFold()

	#Confusion Matrix to get Accuracies
	f1scores = self.getMicroAveragedF1Score(sess,self.caW1,self.caW2,self.caW12,self.caY)

	cm = self.getConfusionMatrix(sess,self.caW1,self.caW2,self.caW12,self.caY)
	print "Total Test Accuracy => ",str(self.getAccuracy(cm))

	cm_train = self.getConfusionMatrix(sess,self.clW1,self.clW2,self.clW12,self.clY)
	print "training accuracy is ",self.getAccuracy(cm_train)

	cm_valid = self.getConfusionMatrix(sess,self.cdW1,self.cdW2,self.cdW12,self.cdY)
	print "validation accuracy is ",self.getAccuracy(cm_valid)

	print "Class Accuracies are "
	print self.getClassesAccuracies(cm)

	print "Micro average f1 measure is "
	print f1scores
		

def mainDima43(flags):
    with tf.Session() as sess:
	print "Welcome to dima 43 classifier"
        nc = NccDiscrepancy(flags)
	print nc.flags.train
        if nc.flags.train: nc.doTrain(sess)
        #nc.doEval(sess)


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
                        help='/path/to/dima43/model')
    parser.add_argument("--train", nargs='?', const=True, default=False, type=bool,
                        help='Train using labeled instances?')
    parser.add_argument("--alltest", nargs='?', const=True, default=False, type=bool,
                        help='Apply model to all instances?')
    parser.add_argument("--resultDir", required=True, type=str,
                        help='/path/to/cascade/model/result/') #default - /data/srijayd/cascade/result/baseline/
    args = parser.parse_args()
    mainDima43(args)


# Accuracy Obtained -> 



