# Implementation of dima classifier

#python ncc_dima43_trimodel.py --indir /Users/srijaydeshpande/IITB_4th_Sem/MTP/glove300 --phrases /Users/srijaydeshpande/IITB_4th_Sem/MTP/f_r1_r2_th/th/th.csv --model /Users/srijaydeshpande/IITB_4th_Sem/MTP/model/dima43_trimodel/ --resultDir /Users/srijaydeshpande/IITB_4th_Sem/MTP/result/ --train true

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
	self.loadThCompoundsCsv(self.phrasesPath)
	self.makeFoldsCompounds()
        logging.info('phrasesDir=%s', self.phrasesDir)
        # first load CSV and split train test
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
		self.allw12.append(0)#commented to ease code, change it to wPhrase when needed
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

    def rankCompoundsNoncomp(self):
    	self.cmpNoncompScores = []
	for i in xrange(self.numDatInst):
		k = 0.5*self.embeds[self.allw1[i]] + 0.5*self.embeds[self.allw2[i]] - self.embeds[self.allw12[i]]
		k=sum(k*k)
		self.cmpNoncompScores.append((k,self.Classes[self.ally[i]]))
	self.cmpNoncompScores.sort(reverse=True)
    
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
	self.clW1, self.clW2, self.clW12, self.clY = [self.allw1[x] for x in le], [self.allw2[x] for x in le], [self.allw12[x] for x in le], [self.ally[x] for x in le]
	self.cdW1, self.cdW2, self.cdW12, self.cdY = [self.allw1[x] for x in de], [self.allw2[x] for x in de], [self.allw12[x] for x in de], [self.ally[x] for x in de]
	self.caW1, self.caW2, self.caW12, self.caY = [self.allw1[x] for x in ap], [self.allw2[x] for x in ap], [self.allw12[x] for x in ap], [self.ally[x] for x in ap]
	print "Folding Finished"

    def getRandomCompMiniBatch(self, batchSize=28):
        sample = np.random.randint(0, len(self.clY), batchSize)
        sWid1 = [self.clW1[x] for x in sample]
        sWid2 = [self.clW2[x] for x in sample]
	sWid12 = [self.clW12[x] for x in sample]
        sY = [self.clY[x] for x in sample]
        return sWid1, sWid2, sWid12, sY

    def buildLossDima43(self):
        self.w1 = tf.placeholder(tf.int32, [None]) #input embedding id 1
        self.w2 = tf.placeholder(tf.int32, [None]) #input embedding id 2
	self.w12 = tf.placeholder(tf.int32, [None]) #input embedding phrase 1_2
        self.y = tf.placeholder(tf.int32, [None])  #output class
        w1v = tf.nn.embedding_lookup(self.embedstf, self.w1) #Embedding table for words
        w2v = tf.nn.embedding_lookup(self.embedstf, self.w2) 
	w12v = tf.nn.embedding_lookup(self.embedstf, self.w12) 
        yh = tf.nn.embedding_lookup(self.oneHot43, self.y) #One hot encodings embedding table for 43 classes
	yh = tf.cast(yh,tf.float32)

	#Neural Network 1
	self.allhidden = self.numDim
	self.hidden1 = 16*self.numDim

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
	pred1_ = tf.map_fn(tf.sigmoid,pred1_)
	pred1 = tf.matmul(pred1_, W1) + b1
	#pred1 = tf.nn.softmax(pred1)

	

	#Neural Network 2

	self.hidden2 = 16*self.numDim

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

	w1v_w12v_norm = tf.sqrt(tf.reduce_sum(tf.square(w1v-w12v), 0, keep_dims=True))
	w2v_w12v_norm = tf.sqrt(tf.reduce_sum(tf.square(w2v-w12v), 0, keep_dims=True))

	#pred2_ = tf.concat(1,[(w1v-w12v)/w1v_w12v_norm,(w2v-w12v)/w2v_w12v_norm]) #Concatenate
	pred2_ = tf.matmul(w2v,Wcomp2) + bcomp2
	pred2_ = tf.map_fn(tf.sigmoid,pred2_)
	pred2 = tf.matmul(pred2_, W2) + b2
	#pred2 = tf.nn.softmax(pred2)


	
	#Neural Network 3

	self.hidden3 = 16*self.numDim

	matSize = self.hidden3 * self.numClasses  
        stddev = 1. / math.sqrt(matSize)
        W3 = tf.Variable(tf.random_normal(shape=[self.hidden3, self.numClasses],mean=0,stddev=stddev))

	matSize = self.numClasses  
        stddev = 1. / math.sqrt(matSize)
	b3 = tf.Variable(tf.random_normal(shape=[self.numClasses],mean=0,stddev=stddev))

	matSize = 2 * self.numDim * self.hidden3  
        stddev = 1. / math.sqrt(matSize)
	Wcomp3 = tf.Variable(tf.random_normal(shape=[2*self.numDim,self.hidden3],mean=0,stddev=stddev))

	matSize = self.hidden3 
        stddev = 1. / math.sqrt(matSize)
	bcomp3 = tf.Variable(tf.random_normal(shape=[self.hidden3],mean=0,stddev=stddev))

        pred3 = tf.concat(1,[w1v,w2v]) #Concatenate

	pred3_ = tf.matmul(pred3,Wcomp3) + bcomp3
	pred3_ = tf.map_fn(tf.sigmoid,pred3_)
	pred3 = tf.matmul(pred3_, W3) + b3
	#pred3 = tf.nn.softmax(pred3)

	self.alpha1 = tf.Variable(tf.random_normal(shape=[1,self.numClasses],mean=0,stddev=1./math.sqrt(self.numClasses)))
	self.alpha2 = tf.Variable(tf.random_normal(shape=[1,self.numClasses],mean=0,stddev=1./math.sqrt(self.numClasses)))
	self.alpha3 = tf.Variable(tf.random_normal(shape=[1,self.numClasses],mean=0,stddev=1./math.sqrt(self.numClasses)))

	pred = self.alpha1*pred1 + self.alpha2*pred2 + self.alpha3*pred3
	#pred = self.alpha3*pred3
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred,yh)
        cross_entropy = tf.reduce_mean(cross_entropy)
        self.loss = cross_entropy
        sgd = tf.train.AdagradOptimizer(0.1)
	#sgd = tf.train.FtrlOptimizer(.2,l1_regularization_strength=0.8,l2_regularization_strength=0.5)
        self.trainop = sgd.minimize(self.loss, var_list=[self.alpha1,self.alpha2,self.alpha3,Wcomp1,bcomp1,W1,b1,Wcomp2,bcomp2,W2,b2,Wcomp3,bcomp3,W3,b3])

	self.predprob = pred #will be used for prediction
	
        self.initop = tf.initialize_all_variables()
        self.saver = tf.train.Saver(tf.trainable_variables())
        logging.info("Built Dima43 network.")

    def doTrain(self, sess, maxIters=100000):
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
	print "Model path is ",tfModelDirPath
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
	sess.run(self.initop, feed_dict={self.place: self.embeds})

        tfModelPath = os.path.join(tfModelDirPath, 'compmodel2.tf')

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
		wid1, wid2, wid12, y = self.getRandomCompMiniBatch()
		_, trloss = sess.run([self.trainop, self.loss],
                                            feed_dict={self.w1: wid1,
                                                       self.w2: wid2,
						       self.w12: wid12,
                                                       self.y: y
                                                       })
            	if xiter % 100 == 0:
                	teloss = sess.run(self.loss,feed_dict={self.w1: self.caW1,
                                                        	self.w2: self.caW2,
						       		self.w12: self.caW12,
                                                        	self.y: self.caY
                                                        	})
			logging.info("Dima43 xiter=%d trloss=%g teloss=%g",xiter, trloss, teloss) #trloss and teloss are numpy ndarrays of one element
			
			#Code for Early stopping for Dima43
		
			currAccuracy = self.get_3nn_accuracy(sess,self.cdW1,self.cdW2,self.cdW12,self.cdY)

			if(currAccuracy > prevAccuracy):
				logging.info("")
				prevAccuracy = currAccuracy
				logging.info("Validation Accuracy => %g",prevAccuracy)
                                testAccuracy = self.get_3nn_accuracy(sess,self.caW1,self.caW2,self.caW12,self.caY)
                                logging.info("Test Accuracy => %g",testAccuracy)
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

    def calClass(self,sess,w1,w2,w12,y):
	probs = sess.run(self.predprob,feed_dict={self.w1: [w1],self.w2: [w2]})
	predClass = np.argmax(probs[0])
	return predClass

    def get_3nn_accuracy(self,sess,w1s,w2s,w12s,ys):
	l = len(ys)
	correctlabels = 0
	predicted = []
	for i in xrange(l):
		predicted.append(self.calClass(sess,w1s[i],w2s[i],w12s[i],ys[i]))
	return recall_score(ys, predicted, average='micro')

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
	print "here precision and recall are"
	print  precision_score(ys, predicted, average='micro')
	print  recall_score(ys, predicted, average='micro')
	return f1_score(ys,predicted, average=None)

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

    def makeResult(self,resultPath,cm,classF1Scores):
	outcmfile = open(resultPath+"result_dima43.csv",'w')
	outcmfile.write("Total Accuracy => ," + str(self.getAccuracy(cm)) + "\n\n")
	outcmfile.write("Class F1 Scores are => ")
	for ind in xrange(len(classF1Scores)):
		outcmfile.write(self.Classes[ind]+","+str(classF1Scores[ind]) + "\n")
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
	outbarchart = resultPath + "barchart_dima43.png" #output png file
	y_pos = np.arange(len(list(self.Classes)))

	assert len(y_pos) == len(classF1Scores)
	plt.bar(y_pos, classF1Scores,width=0.5,alpha=0.5)
	plt.xticks(y_pos,self.Classes,rotation='vertical')
	plt.ylabel('F1-Scores')
	plt.title('Class F1 Scores') 
	plt.show()
	plt.savefig(outbarchart)


    def doEval(self,sess):

    	print "Welcome to Eval"
        tfModelDirPath = os.path.join(self.phrasesDir, self.flags.model)
        if not os.path.isdir(tfModelDirPath): os.mkdir(tfModelDirPath)
        tfModelPath = os.path.join(tfModelDirPath, 'compmodel2.tf')
        #assert os.path.exists(tfModelPath), 'Cannot load ' + tfModelPath
        self.saver.restore(sess, tfModelPath)
        logging.info("Warmstart from %s", tfModelPath)

        #sess.run(self.initop)
	a1 = sess.run(self.alpha1)
	a2 = sess.run(self.alpha2)
	a3 = sess.run(self.alpha3)

	print "alphas are "
	print a1
	print a2
	print a3

	sess.run(tf.initialize_variables([self.embedstf]),feed_dict={self.place: self.embeds})     #For pretrained word embeddings

        #if self.flags.alltest:
            #self.moveTrainToTestFold()

	#Confusion Matrix to get Accuracies
	#f1scores = self.getF1Scores(sess,self.caW1,self.caW2,self.caW12,self.caY)

	cm = self.getConfusionMatrix(sess,self.caW1,self.caW2,self.caW12,self.caY)
	print "Total Test Accuracy => ",str(self.getAccuracy(cm))

	cm_train = self.getConfusionMatrix(sess,self.clW1,self.clW2,self.clW12,self.clY)
	print "training accuracy is ",self.getAccuracy(cm_train)

	cm_valid = self.getConfusionMatrix(sess,self.cdW1,self.cdW2,self.cdW12,self.cdY)
	print "validation accuracy is ",self.getAccuracy(cm_valid)

	print "Class Accuracies are "
	print self.getClassesAccuracies(cm)

	#print "Class F1 Scores are "
	#print f1scores
	
	#self.makeResult(self.flags.resultDir,cm,f1scores)
	

def mainDima43(flags):
    with tf.Session() as sess:
	print "Welcome to dima 43 classifier"
        nc = NccDiscrepancy(flags)
	print nc.flags.train
        if nc.flags.train: nc.doTrain(sess)
        nc.doEval(sess)


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



