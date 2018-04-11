# Implementation of http://www.aclweb.org/anthology/W/W15/W15-0905.pdf
# "Modeling the Statistical Idiosyncrasy of Multiword Expressions".
# I.e., substitution test based on WordNet synonyms alone.

import os, sys, math, argparse, logging, gzip, csv
import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib
from collections import defaultdict

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, logistic, spearmanr
from nltk.corpus import wordnet as wn

from ncc_base import NonComp, NccConst, EvalRecallPrecisionF1


class NccSyn(NonComp):
    def __init__(self, flags):
        super(NccSyn, self).__init__(flags)
        self.phrasesPath = flags.phrases
        self.phrasesDir = os.path.dirname(os.path.realpath(self.phrasesPath))
        logging.info('phrasesDir=%s', self.phrasesDir)
        self.loadPhraseInstancesNcCsv(self.phrasesPath, relax=True)

    def collectGrams(self, w1gramToFreq, w2gramToFreq):
        logging.info("Collecting 1 and 2 grams whose counts we need")
        mod1, mod12 = 0, 0
        for inst in xrange(self.numInst):
            word1 = self.vocabArray[self.allWid1[inst]].word
            word2 = self.vocabArray[self.allWid2[inst]].word
            # TODO(soumenc) Or all senses?
            syn1 = self.mostCommonSense(word1)
            syn2 = self.mostCommonSense(word2)
            nam1 = None if syn1 is None else syn1.lemma_names()
            nam2 = None if syn2 is None else syn2.lemma_names()
            amb1 = nam1 is not None and len(nam1) > 1
            amb2 = nam2 is not None and len(nam2) > 1
            if amb1 and amb2: mod12 += 1
            if not amb1: continue
            mod1 += 1
            w1gramToFreq[word1] = 0
            w1gramToFreq[word2] = 0
            w2gramToFreq[' '.join((word1, word2))] = 0
            for n1 in nam1:
                # note we don't need n1 to be in vocab, as long as it is in ngrams
                w1gramToFreq[n1] = 0
                w2gramToFreq[' '.join((n1, word2))] = 0
        print float(mod1) / self.numInst
        print float(mod12) / self.numInst

    def scanGrams(self, w1gramToFreq, w2gramToFreq):
        logging.info("Scanning to fill %d 1-grams, %d 2-grams",
                     len(w1gramToFreq), len(w2gramToFreq))
        ngramdir = self.flags.ngramdir
        unigramPath = os.path.join(ngramdir, '1gms.csv.gz')
        logging.info("Scanning 1-grams from %s", unigramPath)
        uniLines = 0
        with gzip.open(unigramPath) as unigramFile:
            unigramCsv = csv.reader(unigramFile, delimiter='\t')
            for uniRow in unigramCsv:
                uniLines += 1
                k, v = uniRow[0], uniRow[1]
                if k in w1gramToFreq: w1gramToFreq[k] = int(v)
        logging.info("Scanned %d unigrams", uniLines)
        bigramPath = os.path.join(ngramdir, '2gms.csv.gz')
        logging.info("Scanning 2-grams from %s", bigramPath)
        biLines = 0
        with gzip.open(bigramPath) as bigramFile:
            bigramCsv = csv.reader(bigramFile, delimiter='\t')
            for biRow in bigramCsv:
                biLines += 1
                w1, w2, v = biRow[0], biRow[1], biRow[2]
                k = ' '.join((w1, w2))
                if k in w2gramToFreq: w2gramToFreq[k] = int(v)
        logging.info("Scanned %d 2-grams", biLines)
        # check how many nonzero counts we collected
        zero1, zero2 = 0, 0
        for k, v in w1gramToFreq.iteritems():
            if v == 0:
                logging.debug("missing 1gram [%s]", k)
                zero1 += 1
        for k, v in w2gramToFreq.iteritems():
            if v == 0:
                logging.debug("missing 2gram [%s]", k)
                zero2 += 1
        logging.info("%g%% 1gram %g%% 2gram not found",
                     100. * zero1 / len(w1gramToFreq),
                     100. * zero2 / len(w2gramToFreq))

    def evalFN(self, w1gramToFreq, w2gramToFreq, alpha=15., smoother=.1):
        '''Assume w1gramToFreq w2gramToFreq are primed with available counts.
        Rescan instances and evaluate MWE-ness of phrases.  Here we will only
        try substitutions of mod=w1 and not head=w2.'''
        resp = dict()
        numNoSyn = 0
        for inst in xrange(self.numInst):
            word1 = self.vocabArray[self.allWid1[inst]].word
            word2 = self.vocabArray[self.allWid2[inst]].word
            # TODO(soumenc) Or all senses?
            syn1 = self.mostCommonSense(word1)
            nam1 = None if syn1 is None else syn1.lemma_names()
            amb1 = nam1 is not None and len(nam1) > 1
            if not amb1:
                numNoSyn += 1
                continue  # TODO(soumenc) fix default behavior
            p_w1_w2 = w2gramToFreq[' '.join((word1, word2))]
            p_w1 = w1gramToFreq[word1]
            p_w2_given_w1 = 1. * p_w1_w2 / p_w1
            p_w2_given_sw1_numer, p_w2_given_sw1_denom = 0., 0.
            for w1p in nam1:
                p_w2_given_sw1_numer += w2gramToFreq[' '.join((w1p, word2))]
                p_w2_given_sw1_denom += w1gramToFreq[w1p] + smoother
            p_w2_given_syn_w1 = p_w2_given_sw1_numer / p_w2_given_sw1_denom
            # if you want a binary label:
            # ysys = 1 if p_w2_given_w1 > alpha * p_w2_given_syn_w1 else 0
            # if you want a score, alpha is no longer useful:
            if p_w2_given_syn_w1 == 0:
                ysys = 1.
            else:
                ysys = math.atan(p_w2_given_w1 / p_w2_given_syn_w1) * 2. / np.pi
            resp[','.join((word1, word2))] = [self.allY[inst], ysys]
        return resp

    def run(self):
        w1gramToFreq, w2gramToFreq = defaultdict(int), defaultdict(int)
        self.collectGrams(w1gramToFreq, w2gramToFreq)
        self.scanGrams(w1gramToFreq, w2gramToFreq)
        resp = self.evalFN(w1gramToFreq, w2gramToFreq)
        nccSynPath = os.path.join(self.flags.outdir, NccConst.nccSynName)
        with open(nccSynPath, 'wb') as pickleFile:
            pickle.dump(resp, pickleFile)
        nccEval = EvalRecallPrecisionF1()
        res = nccEval.doEvalDict(resp)
        for k, v in res.iteritems():
            if k != 'recall' and k != 'precision':
                logging.info("result %s %g", k, v)
        _, ax = plt.subplots()
        nccEval.plotRecallPrecision(ax, res['recall'], res['precision'], legend='FN', color='r')
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reload(sys)
    sys.setdefaultencoding('utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True, type=str)
    parser.add_argument("--phrases", required=True, type=str)
    parser.add_argument("--ngramdir", required=True, type=str)
    parser.add_argument("--outdir", required=True, type=str)
    args = parser.parse_args()
    with tf.Session() as sess:
        nccSyn = NccSyn(args)
        nccSyn.run()
