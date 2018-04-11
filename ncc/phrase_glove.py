import os, argparse, sys, logging, heapq, readline
import numpy as np
from ncc_pb2 import VocabWord, Phrase, Token
from seqzfile import SeqzFileWriter
import tensorflow as tf

class IntTrie(object):
    """We may be given a phrase table and have to scan a corpus to identify all/
    maximal occurrences of phrases, rather than a per-document protobuf being given
    to us with all phrases marked already. This class provides a crude trie
     implementation to do that."""
    def __init__(self):
        self.stepMap = dict()
        self.nodeToPid = dict()
        self.nodeGen = -1

    def insert(self, phrase, pid):
        """
        :param phrase: sequence of integers, each a word ID
        :param pid: phrase ID for the whole sequence
        :return:
        """
        node = -1
        for px in phrase:
            if (node, px) in self.stepMap:
                node = self.stepMap[node, px]
            else:
                oldNode = node
                self.nodeGen -= 1
                node = self.nodeGen
                self.stepMap[oldNode, px] = node
        self.nodeToPid[node] = pid

    def logDetailed(self):
        logging.debug(self.stepMap)
        logging.debug(self.nodeToPid)

    def logSummary(self):
        logging.info("Trie has %d steps %d leaves", len(self.stepMap), len(self.nodeToPid))


class PhraseGloveIn(object):
    """Before Glove is trained."""
    def __init__(self, flags):
        self.flags = flags

    def removeCaretFromCorpus(self):
        """Because we use carets to separate words in compounds."""
        with open(self.flags.infile, 'rb') as infile,\
                open(self.flags.outfile, 'wb') as outfile:
            while True:
                ch = infile.read(1)
                if not ch: break
                if ch != '^': outfile.write(ch)

class PhraseGloveOut(object):

    @staticmethod
    def deterministicShuffle(ind, seed=42): #Method copied from phrase_base.py
        """
        :param ind: list to be shuffled
        :param seed: To restore random state.
        :return: ind shuffled in-place.
        """
        randomState = np.random.get_state()
        np.random.seed(seed)
        np.random.shuffle(ind)
        np.random.set_state(randomState)

    """After Glove is trained."""
    def __init__(self, flags):
        logging.info('Begin %s.__init__', 'PhraseGlove')
        self.flags = flags
        #assert self.flags.glove is not None
        self.loadGloveVocab()
        self.loadGloveEmbeddings()
        self.place = tf.placeholder(tf.float32, shape=self.embeds.shape)
        self.embedstf = tf.Variable(self.place, name='embedstf', trainable=False)
        logging.info('End %s.__init__', 'PhraseBase')
	

    def loadGloveVocab(self):
        vocabPath = os.path.join(self.flags.glove, 'vocab.txt')
	logging.info("Loading glove vocabulary from %s",vocabPath)
        with open(vocabPath) as vocabFile:
            self.tidToTerm =  [x.rstrip().split(' ')[0] for x in vocabFile.readlines()]
        #with open(vocabPath) as vocabFile:
            #self.tidToFreq =  [int(x.rstrip().split(' ')[1]) for x in vocabFile.readlines()]  #Commented as it is of no use in running cascade model
        self.termToTid = dict()
        nphrases = 0
        for tid, term in enumerate(self.tidToTerm):
            if '_' in term: 
	    	nphrases += 1
            self.termToTid[term] = tid
	print "Just to check"
	print "id of word the is ",self.termToTid['the']
	print "id of word of is ",self.termToTid['of']
	print "id of phrase recovery_project is ",self.termToTid['recovery_project']
        logging.info('Loaded %d terms from %s', len(self.termToTid), vocabPath)
        logging.info('Found %d phrases out of %d terms', nphrases, len(self.termToTid))

    def loadGloveEmbeddings(self):
        # Scan some lines from text embeddings to find embedding length.
        # Glove text embedding lines do not include the offset.
        embTxtPath = os.path.join(self.flags.glove, 'vectors.txt') #glove.6B.300d.txt
        maxLines, self.dim = 1000, 0
        with open(embTxtPath) as embTxtFile:
            for line in embTxtFile:
                maxLines -= 1
                if maxLines <= 0: break
                ncols = len(line.split(' '))
                assert self.dim == 0 or ncols - 1 == self.dim
                self.dim = ncols - 1
        logging.info('Embeddings in %s have %d elements', embTxtPath, self.dim)
        # Allocate array.
        self.embeds = np.zeros((len(self.termToTid), self.dim), dtype=np.float32)
        logging.info('Embeddings have shape %s', str(self.embeds.shape))
        # Load embeddings into array.
	#self.dim-=1 #This change is only when embeddings are underscored word2vec
        with open(embTxtPath) as embTxtFile:
            for line in embTxtFile:
                cols = line.split(' ')
                term, vec = cols[0], cols[1:]
                if term not in self.termToTid:
                    continue
                tid = self.termToTid[term]
                for cx in xrange(len(vec)): #This change is only when embeddings are underscored word2vec
                    self.embeds[tid, cx] = float(vec[cx])
        logging.info('Loaded embeddings from %s', embTxtPath)

    def act(self):
        """Either store vocabulary and embeddings in NCC format, or test for near
        words and phrases from embeddings loaded into numpy matrix."""
        if args.glove is not None and args.word2vec is not None:
            self.storeWord2vecVocabAndEmbeddings()
        else:
            self.nearTest()

    def nearTest(self, topk=10):
        readline.set_history_length(1000)
        while True:
            query = raw_input('Query word: ')
            if query == '': break
            if query not in self.termToTid: continue
            qid = self.termToTid[query]
            print query, qid
            sims = np.matmul(self.embeds, np.transpose(self.embeds[qid, :]))
            nears = heapq.nlargest(topk, xrange(sims.shape[0]), sims.take)
            print [(x, self.tidToTerm[x]) for x in nears]

    def storeWord2vecVocabAndEmbeddings(self):
        """
        We will retain the token order used by Glove, which mixes words and
        compounds in general.
        """
        vocabMap = dict()
        vocabArray = []
        numWords, numPhrases = 0, 0
        for tid, term in enumerate(self.tidToTerm):
            vw = VocabWord()
            if '_' not in term:
                vw.word = term
                vw.kind = Token
                numWords += 1
            else:
                vw.word = term.replace('_', '^') + '^'
                vw.kind = Phrase
                numPhrases += 1
            assert tid == len(vocabArray)
            vw.wid = len(vocabArray)
            vw.count = self.tidToFreq[tid]
            vocabArray.append(vw)
            vocabMap[vw.word] = vw.wid
        alphaPath = os.path.join(self.flags.word2vec, "alphabet.pbs.gz")
        vrw = SeqzFileWriter(alphaPath)
        for vw in vocabArray:
            vrw.put(vw.SerializeToString())
        vrw.close()
        logging.info('Saved %d words, %d phrases to %s',
                     numWords, numPhrases, alphaPath)
        embedsPath = os.path.join(self.flags.word2vec, "embed.numpy")
        with open(embedsPath, 'wb') as embedsFile:
            self.embeds.tofile(embedsFile)
        logging.info('Saved %s embeddings to %s', str(self.embeds.shape), embedsPath)
        embedsMetaPath = os.path.join(self.flags.word2vec, "embed.meta")
        with open(embedsMetaPath, 'wb') as embedsMetaFile:
            embedsMetaFile.write('{}\n{}\n'.format(self.embeds.shape[0],
                                                 self.embeds.shape[1]))
        logging.info('Saved metadata to %s', embedsMetaPath)

if __name__ == '__main__':
    reload(sys)
    print "Glove Main Started"
    sys.setdefaultencoding('utf-8')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", required=False, type=str,
                        help='/path/to/input/glove/vocab/and/embeddings/')
    parser.add_argument("--word2vec", required=False, type=str,
                        help='/path/to/output/word2vec/vocab/and/embeddings/')
    parser.add_argument("--infile", required=False, type=str,
                        help='/path/to/input/corpus/file')
    parser.add_argument("--outfile", required=False, type=str,
                        help='/path/to/output/corpus/file')
    args = parser.parse_args()
    if args.infile is not None and args.outfile is not None:
        gloveIn = PhraseGloveIn(args)
        gloveIn.removeCaretFromCorpus()
    else:
        gloveOut = PhraseGloveOut(args)
        gloveOut.loadGloveVocab()
        gloveOut.loadGloveEmbeddings()
        gloveOut.act()
