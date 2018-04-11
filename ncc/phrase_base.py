import os, sys, logging, codecs, heapq, argparse, csv
from collections import Counter
import numpy as np
import tensorflow as tf
from seqzfile import SeqzFileReader
from ncc_pb2 import VocabWord, LossKind

class PhraseBase(object):
    @staticmethod
    def sanityCheck(TrainW1_, TrainW2_, TrainLabels_,
                  TestW1_, TestW2_, TestLabels_,
                  DevW1_, DevW2_, DevLabels_):
        """No w12 vectors involved."""
        numTrainInst = np.shape(TrainW1_)[0]
        assert numTrainInst == np.shape(TrainW2_)[0],\
            str(numTrainInst) + '!=' + str(np.shape(TrainW2_)[0])
        assert numTrainInst == np.shape(TrainLabels_)[0]
        numTestInst = np.shape(TestW1_)[0]
        assert numTestInst == np.shape(TestW2_)[0]
        assert numTestInst == np.shape(TestLabels_)[0]
        numDevInst = np.shape(DevW1_)[0]
        assert numDevInst == np.shape(DevW2_)[0]
        assert numDevInst == np.shape(DevLabels_)[0]
        numDim = np.shape(TrainW1_)[1]
        assert numDim == np.shape(TrainW2_)[1]
        assert numDim == np.shape(TestW1_)[1]
        assert numDim == np.shape(DevW1_)[1]
        assert numDim == np.shape(TestW2_)[1]
        assert numDim == np.shape(DevW2_)[1]
        numLabels = np.shape(TrainLabels_)[1]
        assert numLabels == np.shape(TestLabels_)[1]
        assert numLabels == np.shape(DevLabels_)[1]
        return (numDim, numLabels)

    @staticmethod
    def deterministicShuffle(ind, seed=100):
        """
        :param ind: list to be shuffled
        :param seed: To restore random state.
        :return: ind shuffled in-place.
        """
        randomState = np.random.get_state()
        np.random.seed(seed)
        np.random.shuffle(ind)
        np.random.set_state(randomState)

    def __init__(self, flags):
        logging.info('Begin %s.__init__', 'PhraseBase')
        self.flags = flags
        assert self.flags.indir is not None
        #self.loadVocabulary() #-----commented to use stanford glove
        #self.loadEmbeddings() #-----commented to use stanford glove
        self.readStanfordGlove() # to use stanford glove embeddings
        # Embeddings not loaded into TF matrix at init.
        # But we create a variable with a placeholder.
        self.place = tf.placeholder(tf.float32, shape=self.embeds.shape)
        #self.embedstf = tf.Variable(self.place, name='embedstf', trainable=False)
        logging.info('End %s.__init__', 'PhraseBase')

    def readStanfordGlove(self):
        self.vocabArray = []
        self.vocabMap = {}
        self.embeds = []
        infile = open(self.flags.indir,"r")
        i = 0
        for line in infile:
            line = line.split(" ")
	    if(len(line)<=2):
	    	continue
            word = line[0]
            embed = line[1:]
            embed = map(float,embed)
            self.vocabArray.append(word)
            self.vocabMap[word] = i
            i+=1
            self.embeds.append(embed)
        print "All is well"
        self.numDim = len(self.embeds[0])
        self.vocabSize = len(self.vocabArray)
        self.embeds = np.array(self.embeds)    
        print "Embed size is ",self.numDim 
        print "vocabulary size is ",self.vocabSize," and embedsize is ",self.numDim

    def loadVocabulary(self):
        self.vocabMap = { }
        self.vocabSize = 0
        alphaPath = os.path.join(self.flags.indir, "alphabet.pbs.gz")
        logging.info("Loading vocabulary from %s", alphaPath)
        vrr = SeqzFileReader(alphaPath)
	
        while True:
            vocabStr = vrr.get()
            if len(vocabStr)==0: break
            self.vocabSize += 1
	
        vrr.close()
        logging.info("After first pass %d", self.vocabSize)
        self.vocabArray = [ VocabWord() for _ in range(self.vocabSize) ]
        logging.info("Starting second pass.")
        kindCount = Counter()
        vrr = SeqzFileReader(alphaPath)
        counts=[]
        #self.wordToBeExcluded=[]
        while True:
            vocabStr = vrr.get()
            if len(vocabStr)==0: break
            vw = VocabWord()
            vw.ParseFromString(vocabStr)
            #if(vw.count < 100):
                #continue
            self.vocabArray[vw.wid] = vw
            self.vocabMap[vw.word] = vw.wid
            kindCount[vw.kind] += 1
            counts.append(vw.count)
        logging.info("Loaded alphabet %d", self.vocabSize)
        logging.info("KindCounts %s", str(kindCount))

    def countClosedPhrases(self):
        """Given a labeled compound set, count in how many cases the compound
        and both constituents are in our vocabulary."""
        numPhraseProbed, numPhraseFound = 0, 0
        with open(self.flags.phrases) as phFile:
            for phRow in csv.reader(phFile):
                phrase = '^'.join(phRow[0:2]) + '^'
                numPhraseProbed += 1
                if phrase in self.vocabMap:
                    numPhraseFound += 1
                else:
                    logging.debug('Phrase %s not found in vocabulary', phrase)
        logging.info('%d of %d phrases have both words in vocabulary.',
                     numPhraseFound, numPhraseProbed)

    def loadEmbeddings(self):
        """Load from raw numpy file dump into numpy matrix."""
        logging.info("Loading embeddings from %s", self.flags.indir)
        jsonPath = os.path.join(self.flags.indir, 'embed.json')
	#jsonPath = "/data/srijayd/glove50_embed.json"
        with open(jsonPath) as jsonFile:
            import json
            embedMeta = json.load(jsonFile)
            print(embedMeta)
            self.numDim = embedMeta['dim']
            self.vocabSize = embedMeta['vocabSize']
            self.radius = embedMeta['radius']
            self.lossKind = LossKind.Value(embedMeta['lossKind'])
        logging.info('%d %d %g %s', self.numDim, self.vocabSize,
                     self.radius, LossKind.Name(self.lossKind))

        # metaPath = os.path.join(self.flags.indir, 'embed.meta')
        # with open(metaPath) as metaFile:
        #     metaVocabSizeStr, metaDimStr = metaFile.read().split()
        #     metaVocabSize = int(metaVocabSizeStr)
        #     metaDim = int(metaDimStr)
        #     assert metaVocabSize == self.vocabSize
        # self.numDim = metaDim

        numpyPath = os.path.join(self.flags.indir, 'embed.numpy')
        with open(numpyPath, 'rb') as numpyFile:
            self.embeds = np.fromfile(numpyFile, dtype=np.float32,
                              count=self.vocabSize*self.numDim).\
                              reshape((self.vocabSize, self.numDim))
        # 2GB limit makes this fail:
        # self.embedsWrap = tf.constant(self.embeds)
        logging.info("Loaded embeddings from %s and %s, shape=%s",
                     jsonPath, numpyPath, str(self.embeds.shape))
        # Check norms of vectors, rescale to unit norm if flags.
        norms2 = np.linalg.norm(self.embeds, ord=2, axis=1)
        avgNorm = np.average(norms2)
        logging.info("Average embedding vector norm = %g", avgNorm)
        if hasattr(self.flags, 'unit') and self.flags.unit:
            self.embeds = self.embeds / norms2[:,None]
        # Check norms again.
        norms2 = np.linalg.norm(self.embeds, ord=2, axis=1)
        avgNorm = np.average(norms2)
        logging.info("Average embedding vecctor norm = %g", avgNorm)

    def debugPrintEmbeddings(self):
        for tid in xrange(0, 10):
            print self.vocabArray[tid].word,' '
            print self.embeds[tid,:],'\n'

    def loadEmbeddingsTf(self, sess):
        """Copy numpy self.embeds to TF self.embedstf."""
        sess.run(tf.initialize_variables([self.embedstf]),
                 feed_dict={self.place: self.embeds})

    def loadEmbeddingsTf3(self, reuse=None):
        '''Using tf.contrib.embedding.load_embedding(). In case output is marked
        variable, leave out of optimization explicitly.'''
        alphatPath = os.path.join(self.flags.indir, "alphabet.txt")
        assert os.path.exists(alphatPath), alphatPath
        embedsPath = os.path.join(self.flags.indir, "embed.sst")
        assert os.path.exists(embedsPath), embedsPath
    #     embedsHide = tfg.contrib.embedding.load_embedding(
    #         alphatPath, [embedsPath], offset=0, num_entries=self.vocabSize,
    #         embedding_dim=self.numDim,
    #         source_type='TOKEN_EMBEDDING_WITH_SIMPLE_KEY', num_oov_buckets=0)
    #     self.embedstf = tf.Variable(initial_value=embedsHide, trainable=False)
        self.embedLoadOp = tf.contrib.embedding.load_embedding_initializer(
            vocab_file = alphatPath,
            embedding_sources = [embedsPath],
            vocab_size = self.vocabSize,
            embedding_dim = self.numDim,
            source_type = 'TOKEN_EMBEDDING_WITH_SIMPLE_KEY')
        with tf.variable_scope('ncc', reuse=reuse):
            self.embedstf = tf.get_variable('embedstf',
                shape=[self.vocabSize, self.numDim],
                initializer=self.embedLoadOp, trainable=False)
        logging.info("Loaded embeddings via TF: %s", self.embedstf.shape)

    def findNearNumpyDot(self, query, topk):
        if query not in self.vocabMap:
            return []
        qid = self.vocabMap[query]
        print 'nearNumpyDot', query, qid
        sims = np.matmul(self.embeds, np.transpose(self.embeds[qid,:]))
        nears = heapq.nlargest(topk, xrange(sims.shape[0]), sims.take)
        return [(x,self.vocabArray[x].word) for x in nears]

    def findNearNumpyL2(self, query, topk):
        if query not in self.vocabMap:
            return []
        qid = self.vocabMap[query]
        print 'nearNumpyL2', query, qid
        diffVecs = self.embeds - self.embeds[qid, :]
        l2s = np.sum(np.multiply(diffVecs, diffVecs), axis=1)
        nears = heapq.nsmallest(topk, xrange(l2s.shape[0]), l2s.take)
        return [(x, self.vocabArray[x].word) for x in nears]

    def findNearNumpy(self, query, topk=10):
        if self.lossKind == LossKind.Value('Dot'):
            return self.findNearNumpyDot(query, topk)
        elif self.lossKind == LossKind.Value('L2'):
            return self.findNearNumpyL2(query, topk)
    
    def findNearTfDot(self, query, sess, topk=10):
        """Given a vocab wid, find wids nearest to it.
        Uses embedding stored in TF var/const."""
        #sims = np.dot(self.embeds, np.transpose(self.embeds[wid,:]))
        if query not in self.vocabMap:
            return []
        qid = self.vocabMap[query]
        print query, qid
        simstf = tf.matmul(self.embedstf,
                           tf.expand_dims(self.embedstf[qid,:], 0),
                           transpose_b = True)
        sims = sess.run([simstf])[0]
        assert sims.shape[0] == self.embeds.shape[0]
        nears = heapq.nlargest(topk, xrange(sims.shape[0]), sims.take)
        return [(x, self.vocabArray[x].word) for x in nears]

    def storeVocabularyAsText(self):
        """
        Store vocabulary to text file for viewing.
        No longer used by tf.contrib.embedding.load_embedding.
        """
        alphatPath = os.path.join(self.flags.indir, "alphabet.txt")
        with codecs.open(alphatPath, "w", 'UTF-8') as alphatFile:
            alphaCsvw = csv.writer(alphatFile, delimiter='\t')
            for vw in self.vocabArray:
                alphaCsvw.writerow((vw.word, vw.wid, vw.count, vw.kind))

# Test harness, finds near neighbors to query words/phrases.
if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True, type=str)
    parser.add_argument("--unit", required=False, action='store_true')
    args = parser.parse_args()
    pb = PhraseBase(args)
    with tf.Session() as sess:
        pb.loadEmbeddingsTf(sess)
        # pb.debugPrintEmbeddings()
        while True:
            query = raw_input('Query word: ')
            if query == '': break
            #print pb.findNearTfDot(query, sess)
            print pb.findNearNumpy(query)
