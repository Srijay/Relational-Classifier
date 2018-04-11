import os, sys, math, csv, re, logging
import cPickle as pickle
import numpy as np
from numpy.lib.shape_base import row_stack
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr
from nltk.corpus import wordnet as wn
from operator import itemgetter

import tensorflow as tf
from phrase_base import PhraseBase


class NccConst(object):
    """Static constant fields."""

    modFvCsvName = 'modfv.csv'
    """CSV file where alternative modifier feature vectors are saved in
    an easily readable format."""

    modFvPickleName = 'modfv.pickle'
    """The CSV file is not easily parsed, so we also save a pickle file
    with a sequence of ModFv records."""

    ngramPickleName = 'ngram.pickle'
    """Pickle file where ngram digests are stored."""

    substPickleName = 'subst.pickle'
    """Pickle file where per-phrase gold and test scores from
    substitution test are stored."""

    nccSynName = 'syn.pickle'
    """Farahmand-Nivre synset substitution scores per phrase."""


class NonComp(PhraseBase):
    def __init__(self, flags):
        logging.info('Begin %s.__init__', 'NonComp')
        super(NonComp, self).__init__(flags)
        self.width = 50
        # We do not load any labeled instances nor do train-dev-test split here.
        logging.info('End %s.__init__', NonComp.__name__)

    def loadPhraseInstancesThCsv(self, thCsvPath):
        """Load labeled instances from TH-style CSV file and fill arrays.
        Need to convert from 43-way to 2-way labels."""
        self.allWid1, self.allWid2, self.allWid12, self.allY = [], [], [], []
        ncLabels = set(['Other.MWE', ])
        numRows, numComplete = 0, 0
        with open(thCsvPath, mode='r') as th:
            for row in csv.reader(th, delimiter=','):
                y = 1 if (row[2] in ncLabels) else 0
                # We need both word IDs from vocab, but phrase ID may be missing
                iv1, iv2 = row[0] in self.vocabMap, row[1] in self.vocabMap
                if not iv1 or not iv2: continue
                wMod, wHead = self.vocabMap[row[0]], self.vocabMap[row[1]]
                sPhrase = "^".join([row[0], row[1], ''])
                iv12 = sPhrase in self.vocabMap
                if iv12:
                    wPhrase = self.vocabMap[sPhrase]
                    self.allWid1.append(wMod)
                    self.allWid2.append(wHead)
                    self.allWid12.append(wPhrase)
                    self.allY.append(y)
                    numComplete += 1
                numRows += 1
        numInst = self.numInst = len(self.allY)
        assert numInst == len(self.allWid1)
        assert numInst == len(self.allWid2)
        assert numInst == len(self.allWid12)
        logging.info("Loaded %d rows from %s, %d complete",
                     numRows, thCsvPath, numComplete)

    def loadPhraseInstancesNcCsv(self, ncCsvPath, relax=False):
        """Load word pairs and labels from multi-editor noncomp judgment files.
        relax=True means phrase need not be in vocab.
        Unlabeled instances are allowed and assigned label = -1."""
        self.allWid1 = []
        self.allWid2 = []
        self.allWid12 = []
	self.literal1 = []
	self.literal2 = []
        self.allY = []
	self.Mod = []
	self.Head = []
        numRows, numWordsKnown, numPhraseKnown, numLabeled = 0, 0, 0, 0
        with open(ncCsvPath, mode='r') as nc:
            for row in csv.reader(nc, delimiter=','):
                assert len(row) >= 2, row
                numRows += 1
                iv1, iv2 = row[0] in self.vocabMap, row[1] in self.vocabMap
                if not iv1 or not iv2: continue
                numWordsKnown += 1
                wMod, wHead = self.vocabMap[row[0]], self.vocabMap[row[1]]
		self.Mod.append(row[0])
		self.Head.append(row[1])
                sPhrase = "^".join([row[0], row[1], ''])
                hasLabel = len(row) > 2
                if hasLabel:
                    numLabeled += 1
                    try:
                        nrow = np.array(row[2], dtype="float32") #change is done to allow literals of reddys
                    except ValueError:
                        logging.fatal(row)
                    y = np.average(nrow)
                else:
                    y = -1  # TODO(soumenc) review
                iv12 = sPhrase in self.vocabMap
                if iv12: numPhraseKnown += 1
                iv12relax = relax or iv12
                if iv12relax:
                    wPhrase = self.vocabMap[sPhrase] if iv12 else -1  # TODO(soumenc) review
                    self.allWid1.append(wMod)
                    self.allWid2.append(wHead)
                    self.allWid12.append(wPhrase)
		    #self.literal1.append(float(row[3]))
		    #self.literal2.append(float(row[4]))
                    self.allY.append(y)
        numInst = self.numInst = len(self.allY)
        assert numInst == len(self.allWid1)
        assert numInst == len(self.allWid2)
        assert numInst == len(self.allWid12)
        logging.info("Loaded from %s: rows=%d wordsKnown=%d phraseKnown=%d labeled=%d",
                     ncCsvPath, numRows, numWordsKnown, numPhraseKnown, numLabeled)

    def trainTestSplit(self):
        """Approx 1/2-1/2 in train and test folds.  Nothing in dev fold.
        For determinism, we set the seed and restore the old one when done."""
        ind = np.arange(self.numInst)
        self.deterministicShuffle(ind)
        tr, te = np.array_split(ind, 2)
        dv = []
        self.trainDevTestCollect(tr, dv, te)

    def trainDevTestSplit(self):
        """Approx 1/3rd in train, dev, test folds.
        For determinism, we set the seed and restore the old one when done."""
        ind = np.arange(self.numInst)
        self.deterministicShuffle(ind)
        tr, dv, te = np.array_split(ind, 3)
        self.trainDevTestCollect(tr, dv, te)

    def trainDevTestCollect(self, tr, dv, te):
        logging.info("Collect train fold %d", len(tr))
        self.trWid1, self.trWid2, self.trWid12, self.trY = \
            [self.allWid1[x] for x in tr], [self.allWid2[x] for x in tr], \
            [self.allWid12[x] for x in tr], [self.allY[x] for x in tr]
        logging.info("Collect dev fold %d", len(dv))
        self.dvWid1, self.dvWid2, self.dvWid12, self.dvY = \
            [self.allWid1[x] for x in dv], [self.allWid2[x] for x in dv], \
            [self.allWid12[x] for x in dv], [self.allY[x] for x in dv]
        logging.info("Collect test fold %d", len(te))
        self.teWid1, self.teWid2, self.teWid12, self.teY = \
            [self.allWid1[x] for x in te], [self.allWid2[x] for x in te], \
            [self.allWid12[x] for x in te], [self.allY[x] for x in te]
        logging.info("Train dev test splits %d %d %d", len(self.trWid1),
                     len(self.dvWid12), len(self.teY))


    def mostCommonSense(self, word):
        """ http://stackoverflow.com/questions/5928704/how-do-i-find-the-frequency-count-of-a-word-in-english-using-wordnet """
        syns = wn.synsets(word, pos=wn.NOUN)
        freqs = []
        for syn in syns:
            freq = 0
            for lemma in syn.lemmas(): freq += lemma.count()
            freqs.append(freq)
        if len(syns) == 0: return None
        argmax, _ = max(enumerate(freqs), key=itemgetter(1))
        return syns[argmax]


    def mostCommonSenseSimilarity(self, word1, word2):
        """Returns NLTK path similarity, 0 if that is None."""
        syn1, syn2 = self.mostCommonSense(word1), self.mostCommonSense(word2)
        if syn1 is None or syn2 is None: return 0
        ans = wn.path_similarity(syn1, syn2)
        return 0 if ans is None else ans


    def mostCommonSenseSimilarities(self, word1, word2):
        """Returns a vector with several wn-based similarities.
        Any of the similarity scores can be None."""
        syn1, syn2 = self.mostCommonSense(word1), self.mostCommonSense(word2)
        if syn1 is None or syn2 is None: return [0., 0., 0.]
        return [syn1.path_similarity(syn2),
                syn1.lch_similarity(syn2),
                syn1.wup_similarity(syn2)]


    def buildVariable(self, shape, stddev, name):
        return tf.Variable(tf.random_normal(
            shape=shape, mean=0, stddev=stddev), name=name)


    def getRandomMiniBatch(self, batchSize):
        """Sample a batch from training fold."""
        sample = np.random.randint(0, len(self.trY), batchSize)
        sWid1 = [self.trWid1[x] for x in sample]
        sWid2 = [self.trWid2[x] for x in sample]
        sWid12 = [self.trWid12[x] for x in sample]
        sY = [self.trY[x] for x in sample]
        return sWid1, sWid2, sWid12, sY


    def dumpTfGraph(self):
        tfg = open("/tmp/tfgraph.txt", "w")
        tfg.write(str(tf.get_default_graph().as_graph_def()))
        tfg.close()
        print>> sys.stderr, "Dumped tf.graph"


class EvalRecallPrecisionF1(object):
    """Given noncomp judgment and system output for each phrase,
    compute data for recall precision graph and max F1 over all ranks.
    Number of instances must be small enough to fit labels in RAM."""

    def __init__(self, sortOrderDecr=False):
        """Sort decreasing by system score?"""
        self.sortOrderDecr = sortOrderDecr

    def doEvalList(self, ygold, ysys, ncThresh=.5):
        """
        Inputs:
         ygold is np.array noncomp gold judgment in [0,1], 1 = fully noncomp.
         ysys is np.array system score, arbitrary real.  By default, sort order is
         increasing. The assumption being, larger the system score, more the
         belief that the compound is non compositional (e.g., L2 distance between
         estimated and observed phrase embeddings).  If instead we take dot product,
         smaller system score will imply more confidence in non compositionality,
         in which case, sort order should be specified as decreasing in the constructor.
        Outputs:
         recall at all ranks
         precision at all ranks
         max F1 over all ranks
         Spearman's rho
         Kendall's tau-b
         Pearson r (despite the broken normality assumption)
        """
        assert len(ygold) == len(ysys), '{} != {}'.format(len(ygold), len(ysys))
        glen = len(ygold)
        perm = ysys.argsort() if self.sortOrderDecr else ysys.argsort()[::-1][:glen]
        logging.debug('System %s', str(ysys[perm]))
        logging.debug('Gold %s', str(ygold[perm]))
        bgold = [(yg >= ncThresh) for yg in ygold]  # gold turned boolean
        ngold = [(1. if bg else 0.) for bg in bgold]  # boolean to 0/1
        suffixFn = np.sum(ngold)
        tp, fp, fn = np.zeros(glen), np.zeros(glen), np.zeros(glen)
        for inst in xrange(glen):
            # we just marked inst with label = 1
            if bgold[perm[inst]]:
                tp[inst] = 1
                suffixFn -= 1
            else:
                fp[inst] = 1
            fn[inst] = suffixFn
            if inst > 0:
                tp[inst] += tp[inst - 1]
                fp[inst] += fp[inst - 1]
        recall = 1. * tp / (tp + fn)
        precision = 1. * tp / (tp + fp)
        f1 = 2. * recall * precision / (recall + precision)
        f1z = [0 if np.isnan(f_) else f_ for f_ in f1]
        rho, _ = spearmanr(ygold, ysys)
        ktau = kendalltau(ygold, ysys)
        pear = pearsonr(ygold, ysys)
        ndcg1 = self.getNdcg1(bgold, ngold, perm)
        ndcg2 = self.getNdcg2(bgold, ngold, perm)
        p100 = self.getP100(bgold, perm)
        return {'recall': recall, 'precision': precision,
                'p100': p100, 'maxf1': max(f1z),
                'ndcg1': ndcg1, 'ndcg2': ndcg2,
                'rho': rho, 'ktau': ktau.correlation,
                'pearson': pear[0] }

    def getP100(self, bgold, perm):
        """P@100; if 100 not available, extrapolate."""
        numRank, numNc = 0., 0.
        for ix in xrange(100):
            if ix >= len(perm): break
            numRank += 1.
            if bgold[perm[ix]]: numNc += 1.
        p100 = numNc * 100. / numRank
        return p100

    def getNdcg1(self, bgold, ngold, perm):
        """First definition in Wikipedia."""
        dcg, idcg = 0., 0.
        numNc = int(0)
        for rank in xrange(len(perm)):
            if bgold[perm[rank]]:
                numNc += 1
                if rank == 0:
                    dcg += 1.
                else:
                    dcg += 1. / math.log(1. + rank, 2.)
        for rank in xrange(numNc):
            if rank == 0:
                idcg += 1.
            else:
                idcg += 1. / math.log(1. + rank, 2.)
        ndcg = dcg / idcg
        assert 0. <= ndcg and ndcg <= 1., ndcg
        return ndcg

    def getNdcg2(self, bgold, ngold, perm):
        """Second definition in Wikipedia, more commonly used."""
        dcg, idcg = 0., 0.
        for rank in xrange(len(perm)):
            dcg += ngold[perm[rank]] / math.log(2. + rank)
        numRel = int(np.sum(ngold))
        for rank in xrange(numRel):
            idcg += 1. / math.log(2. + rank)
        ndcg = dcg / idcg
        assert 0. <= ndcg and ndcg <= 1., ndcg
        return ndcg

    def doEvalDict(self, xToYs):
        """Wrapper around method with inputs ygold, ysys."""
        ygold = np.zeros(len(xToYs), dtype='float32')
        ysys = np.zeros(len(xToYs), dtype='float32')
        ix = 0
        for x in sorted(xToYs.keys()):
            ygold[ix] = xToYs[x][0]
            ysys[ix] = xToYs[x][1]
            ix += 1
        return self.doEvalList(ygold, ysys)

    def plotRecallPrecision(self, ax, recall, precision,
                            legend, color, marker='o'):
        """Adds one recall-precision plot to a chart."""
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.grid(True)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.scatter(recall, precision, s=2, label=legend, c=color,
                   facecolors='none', edgecolors=color, marker=marker, lw=0.5)
