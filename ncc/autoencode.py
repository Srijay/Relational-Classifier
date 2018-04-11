import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path

infile = open('cluster_data.txt','r')

points = []

for line in infile:
    pt = line.split()
    pt = map(float,pt)
    points.append(pt)
    
batchSize = 30
encodeSize = 2
inputSize = len(points[0])
maxIter = 100000
modelpath = "/Users/srijaydeshpande/MyMLApps/neural_clustering/model"

def generateSamplesToAutoEncode(points):
    sample = np.random.randint(0,len(points),batchSize)
    retdata = [points[x] for x in sample]
    return retdata


def autoEncode(points, batchSize, encodeSize, maxIter, modelpath):
    input = tf.placeholder(tf.float32, shape=[None,inputSize])

    W_in = tf.Variable(tf.random_normal(shape=[inputSize,encodeSize], mean=0.3, stddev=0.1))
    b_in = tf.Variable(tf.random_normal(shape=[encodeSize], mean=0, stddev=0.1))
    W_out = tf.Variable(tf.random_normal(shape=[encodeSize,inputSize], mean=0.3, stddev=0.1))
    b_out = tf.Variable(tf.random_normal(shape=[inputSize], mean=0, stddev=0.1))

    tfencoded = tf.matmul(input,W_in) + b_in
    reconstructed = tf.matmul(tfencoded,W_out) + b_out

    recon_loss = tf.reduce_sum(pow((input-reconstructed),2))

    trainop = tf.train.AdamOptimizer(.01).minimize(recon_loss)
    initop = tf.initialize_all_variables()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(initop)
    
    if(os.path.exists(modelpath+".index")):
        print("Restoring the model")
        saver.restore(sess,modelpath)
        print("Model restored.")
    else:
        
        print("Training Starts")

        for i in xrange(maxIter):
            traindata = generateSamplesToAutoEncode(points)    
            _,trainloss,trinput,trreconstructed = sess.run([trainop,recon_loss,W_in,W_out],feed_dict={input:traindata})
            if(i%1000==0):
                print(trainloss) 
            
        save_path = saver.save(sess,modelpath) 
        print("Model saved in path: %s" % save_path)
    
    encodes = sess.run(tfencoded,feed_dict={input:points})

    return encodes
    
encoded = autoEncode(points,batchSize,encodeSize,maxIter,modelpath)

x = []
y=[]
labels = []
i=0

for pt in points:
    if(pt[0] > 0):
        labels.append(0)
    elif(pt[1] > 0):
        labels.append(1)
    else:
        labels.append(2)
    encode = encoded[i]
    i+=1
    x.append(encode[0])
    y.append(encode[1])

colors = ['red','green','blue']

fig = plt.figure(figsize=(8,8))
plt.scatter(x, y, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()










    