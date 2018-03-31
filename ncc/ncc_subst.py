# Generalized substitution test for (non)compositionality.

import os, sys, math, csv, logging, gzip, argparse
import numpy as np
import cPickle as pickle
import matplotlib

matplotlib.use('TkAgg')
import tensorflow as tf

from collections import Counter, defaultdict
from os.path import expanduser
from scipy.stats import gaussian_kde, logistic, spearmanr
from matplotlib import pyplot as plt

from ncc_base import NccConst, NonComp, EvalRecallPrecisionF1
from ncc_pb2 import VocabWord


class ModFv(object):
    """
    For each labeled phrase instance, records sibling modifiers and their 
    feature vectors.  A sequence of ModFv records are pickled into a file
    for easy exploratory analysis. 
    """

    def __init__(self, sMod, sHead, y, pmiModHead):
        """All alternative modifiers for one (mod, head) pair."""
        self.sMod, self.sHead, self.y = sMod, sHead, y
        self.pmiModhead = pmiModHead
        # TODO(soumenc) Better to know the number of alts and create np.matrix?
        self.sAltMods = []
        self.altModFvs = []

    def appendAltMod(self, sAltMod, altModFv):
        """sAltMod is the string form of the alternative modifier
        altModFv is a np.array feature vector of the alternative modifier"""
        if sAltMod == self.sMod: return
        self.sAltMods.append(sAltMod)
        self.altModFvs.append(altModFv)


class NccSubstModel(object):
    """Model weights to multiply with modifier feature vector.
    [[0.8], [0.1], [0.3], [0.3], [0.1]];0.1;1;0.5 0.416667 40 0.773768
    [[0.9], [0.3], [0.2], [0.3], [0.1]];0.05;2.0;0.5 0.416048 39 0.813739
    """

    def __init__(self, fw=[.9, .3, .2, .3, .1], fb=.05, nss=2., nsb=.5):
        self.featureWeights = np.matrix(fw).transpose()
        self.featureBias = fb
        self.numSoftModsScale = nss
        self.numSoftModsBias = nsb

    def __str__(self):
        return str(self.featureWeights.tolist()) + ';' + str(self.featureBias) \
               + ';' + str(self.numSoftModsScale) + ';' + str(self.numSoftModsBias)

    def prediction(self, modfv):
        """For each altMod, multiples its feature vector with featureWeights,
        subtracts featureBias, applies sigmoid, returns sum."""
        assert len(modfv.sAltMods) == len(modfv.altModFvs)
        numSoftMods = 0
        for alt in xrange(len(modfv.sAltMods)):
            dot = np.asscalar(np.dot(modfv.altModFvs[alt], self.featureWeights))
            numSoftMods += logistic.cdf(dot - self.featureBias)
        # 1 - ... because we need to output noncomp confidence.
        return (1. - logistic.cdf(numSoftMods / self.numSoftModsScale
                                  - self.numSoftModsBias))


class NccSubstPrepare(NonComp):
    """
    Preloads the following tables from 1gram and 2gram count sstables,
    after intersecting with (mod, head) from a given set of phrases:
      * head-to-mod-to-count
      * mod-to-head-to-count
      * unigram counts for all head and mods appearing in above
      * sum of all 1gram and 2gram counts from sstables.
    """

    def __init__(self, flags):
        super(NccSubstPrepare, self).__init__(flags)
        self.phrasesPath = flags.phrases
        self.phrasesDir = os.path.dirname(os.path.realpath(self.phrasesPath))
        logging.info('phrasesDir=%s', self.phrasesDir);
        self.loadPhraseInstancesNcCsv(self.phrasesPath, relax=True)
        if not self.loadOtherWordStats(): self.saveOtherWordStats()
        # Diagnostics counters for logging 
        # average substituent modifiers (heads) for each head (modifier).
        inHeadSumMods, inModSumHeads = long(0), long(0)
        for _, modCounts in self.headToModifierStats.iteritems():
            inHeadSumMods += len(modCounts)
        for _, headCounts in self.modifierToHeadStats.iteritems():
            inModSumHeads += len(headCounts)
        logging.info("avgHeadFanout=%g avgModFanout=%g",
                     float(inHeadSumMods) / len(self.headToModifierStats),
                     float(inModSumHeads) / len(self.modifierToHeadStats))
        logging.info("NccSubstPrepare.__init__")

    def loadOtherWordStats(self):
        """If self.saveOtherWordStats has run and saved its output, then load it. """
        ngramPicklePath = os.path.join(self.phrasesDir, NccConst.ngramPickleName)
        if not os.path.isfile(ngramPicklePath): return False
        logging.info("Will load ngram digest from %s.", ngramPicklePath)
        with open(ngramPicklePath, 'rb') as ngramPickleFile:
            (self.modifierToHeadStats,
             self.headToModifierStats,
             self.unigramCounts,
             self.sum1gramCounts,
             self.sum2gramCounts) = pickle.load(ngramPickleFile)
        logging.info("Loaded pickled ngram digest.")
        return True

    def unigrams(self, unigramPath):
        with gzip.open(unigramPath) as unigramFile:
            unigramCsv = csv.reader(unigramFile, delimiter='\t')
            for uniRow in unigramCsv:
                yield uniRow[0], uniRow[1]

    def bigrams(self, bigramPath):
        with gzip.open(bigramPath, 'rb') as bigramFile:
            bigramCsv = csv.reader(bigramFile, delimiter='\t')
            for biRow in bigramCsv:
                yield biRow[0], biRow[1], biRow[2]

    def saveOtherWordStats(self):
        """ Too slow to probe ngram sstables in random order, so we create dicts
        from the phrase instances and fill them via sequential scan. """
        foundUnigramCount = dict()
        foundBigramCount = dict()
        for inst in range(self.numInst):
            w1, w2 = self.allWid1[inst], self.allWid2[inst]
            word1, word2 = self.vocabArray[w1].word, self.vocabArray[w2].word
            foundUnigramCount[word1] = 0
            foundUnigramCount[word2] = 0
            foundBigramCount[word1 + ' ' + word2] = 0  # ngram uses space sep
        ngramdir = self.flags.ngramdir
        uniPath = os.path.join(ngramdir, '1gms.csv.gz')
        logging.info("Scanning 1-grams from %s", uniPath)
        uniLines = 0
        for k, v in self.unigrams(uniPath):
            uniLines += 1
            if k in foundUnigramCount: foundUnigramCount[k] = v
        # with gzip.open(uniPath) as unigramFile:
        #     unigramCsv = csv.reader(unigramFile, delimiter='\t')
        #     for uniRow in unigramCsv:
        #         uniLines += 1
        #         k, v = uniRow[0], uniRow[1]
        #         if k in foundUnigramCount: foundUnigramCount[k] = v
        logging.info("Scanned %d unigrams", uniLines)
        bigramPath = os.path.join(ngramdir, '2gms.csv.gz')
        logging.info("Scanning 2-grams from %s", bigramPath)
        biLines = 0
        for w1, w2, v in self.bigrams(bigramPath):
            biLines += 1
            k = ' '.join((w1, w2))
            if k in foundBigramCount: foundBigramCount[k] = v
        # with gzip.open(bigramPath) as bigramFile:
        #     bigramCsv = csv.reader(bigramFile, delimiter='\t')
        #     for biRow in bigramCsv:
        #         biLines += 1
        #         w1, w2, v = biRow[0], biRow[1], biRow[2]
        #         k = ' '.join((w1, w2))
        #         if k in foundBigramCount: foundBigramCount[k] = v
        logging.info("Scanned %d 2-grams", biLines)
        self.sum1gramCounts = long(0)
        self.sum2gramCounts = long(0)
        # These are dict with key = word ID, val = Counter()
        self.headToModifierStats = dict()
        self.modifierToHeadStats = dict()
        logging.info("Starting saveOtherWordStats")
        # This has  unigram counts for all word IDs present in above.
        self.unigramCounts = Counter()
        n1lookup, n1found, n2lookup, n2found = 0, 0, 0, 0
        for inst in xrange(self.numInst):
            w1, w2 = self.allWid1[inst], self.allWid2[inst]
            word1, word2 = self.vocabArray[w1].word, self.vocabArray[w2].word
            found1, found2 = word1 in foundUnigramCount, word2 in foundUnigramCount
            found12 = word1 + ' ' + word2 in foundBigramCount
            n1lookup = n1lookup + 2
            n2lookup = n2lookup + 1
            if found1: n1found = n1found + 1
            if found2: n1found = n1found + 1
            if found12: n2found = n2found + 1
            if found1 and found2 and found12:
                self.headToModifierStats[w2] = Counter()
                self.modifierToHeadStats[w1] = Counter()
        logging.info("Initialized headToModifierStats and modifierToHeadStats.")
        for modWord, headWord, phraseCount in self.bigrams(bigramPath):
            self.sum2gramCounts += long(phraseCount)
            # TODO(soumenc@) TODO(inaim@) Bump count before vocab check?
            if modWord not in self.vocabMap: continue
            if headWord not in self.vocabMap: continue
            modWid, headWid = self.vocabMap[modWord], self.vocabMap[headWord]
            if modWid in self.modifierToHeadStats:
                self.modifierToHeadStats[modWid][headWid] += int(phraseCount)
            if headWid in self.headToModifierStats:
                self.headToModifierStats[headWid][modWid] += int(phraseCount)
            self.unigramCounts[headWid] = 0
            self.unigramCounts[modWid] = 0
        logging.info("Initialized 2-gram counts; 1-gram slots=%d",
                     len(self.unigramCounts))
        for oWord, oCount in self.unigrams(uniPath):
            self.sum1gramCounts += long(oCount)
            # TODO(soumenc@) TODO(inaim@) Bump count before vocab check? 
            if oWord not in self.vocabMap: continue
            oWid = self.vocabMap[oWord]
            if oWid in self.unigramCounts:
                self.unigramCounts[oWid] = long(oCount)
        logging.info("Initialized 1-gram counts, sum=%g", self.sum1gramCounts)
        ngramPicklePath = os.path.join(self.flags.outdir, NccConst.ngramPickleName)
        with open(ngramPicklePath, mode="wb") as ngramPickleFile:
            pickle.dump((self.modifierToHeadStats,
                         self.headToModifierStats,
                         self.unigramCounts,
                         self.sum1gramCounts,
                         self.sum2gramCounts),
                        ngramPickleFile)
        logging.info("Saved pickled maps to %s", ngramPicklePath)
        print>> sys.stderr, n1found, '/', n1lookup, ', ', n2found, '/', n2lookup, \
            ' ...', len(self.headToModifierStats), ',', len(self.modifierToHeadStats)

    # TODO(soumenc) Make a flag?
    def countFilter(self, cx):
        return cx >= 5

    # TODO(soumenc) Make a flag?
    def word2vecDotFilter(self, dot):
        return dot >= 0.6

    # TODO(soumenc) Make a flag? Must be a positive threshold.
    def pmiFilter(self, pmi):
        return pmi >= 0.1

    # TODO(soumenc) TODO(inaim) Replace stub with test for relatedness.
    def wordnetFilter(self, word1, word1o):
        return True

    def getPmi(self, wMod, wHead):
        """TODO(soumenc) TODO(inaim) review "return 0" situations."""
        # marginals
        if wMod not in self.unigramCounts: return 0
        if wHead not in self.unigramCounts: return 0
        pMod = 1. * self.unigramCounts[wMod] / self.sum1gramCounts
        pHead = 1. * self.unigramCounts[wHead] / self.sum1gramCounts
        # joint
        if wHead not in self.headToModifierStats: return 0
        wOtherModCounts = self.headToModifierStats[wHead]
        if wMod not in wOtherModCounts: return 0
        pModHead = 1. * wOtherModCounts[wMod] / self.sum2gramCounts
        return math.log(pModHead) - math.log(pMod) - math.log(pHead)

    def saveModFv(self, force=False):
        modFvPicklePath = os.path.join(self.flags.outdir, NccConst.modFvPickleName)
        modFvCsvPath = os.path.join(self.flags.outdir, NccConst.modFvCsvName)
        exist = os.path.exists(modFvPicklePath) and os.path.exists(modFvCsvPath)
        if exist and not force: return
        with open(modFvCsvPath, 'wb') as modFvCsvFile, \
                open(modFvPicklePath, 'wb') as modFvPickleFile:
            modFvCsv = csv.writer(modFvCsvFile, delimiter='\t',
                                  quotechar='|', quoting=csv.QUOTE_MINIMAL)
            modFvCsv.writerow(['Alt/Mod', 'Head', 'Pmi/Ratio', 'Wn', 'Cos'])
            numInstAccepted = 0
            numSubstConsidered, numSubstRejectedByCount = 0, 0
            numSiblingToFreq = Counter()
            for inst in xrange(self.numInst):
                if inst % 1000 == 0:
                    logging.info("%d of %d instances processed.", inst, self.numInst)
                    logging.info("%d inst accepted; %d of %d siblings pruned by count",
                                 numInstAccepted, numSubstRejectedByCount, numSubstConsidered)
                wMod, wHead, y = self.allWid1[inst], self.allWid2[inst], self.allY[inst]
                if wHead not in self.headToModifierStats: continue
                wOtherModCounts = self.headToModifierStats[wHead]
                sMod, sHead = self.vocabArray[wMod].word, self.vocabArray[wHead].word
                vMod, vHead = self.embeds[wMod, :], self.embeds[wHead, :]
                pmiModHead = self.getPmi(wMod, wHead)
                if not self.pmiFilter(pmiModHead): continue
                modFvCsv.writerow([])
                modFvCsv.writerow([sMod, sHead, y, pmiModHead])
                numInstAccepted += 1
                numSiblingFiltered = 0
                modFv = ModFv(sMod, sHead, y, pmiModHead)
                numSubstConsidered += len(wOtherModCounts)
                logging.info("(%s %s) --> %d", sMod, sHead, len(wOtherModCounts))
                for wOtherMod, wOtherCount in wOtherModCounts.iteritems():
                    if not self.countFilter(wOtherCount):
                        numSubstRejectedByCount += 1
                        continue
                    sOtherMod = self.vocabArray[wOtherMod].word
                    vOtherMod = self.embeds[wOtherMod, :]
                    dotV1V1o = np.dot(vMod, vOtherMod)
                    if not self.word2vecDotFilter(dotV1V1o): continue
                    pmiW1oW2 = self.getPmi(wOtherMod, wHead)
                    pmiRatio = pmiW1oW2 / pmiModHead
                    if not self.pmiFilter(pmiW1oW2): continue
                    wnSim = self.mostCommonSenseSimilarity(sMod, sOtherMod)
                    wnSims = self.mostCommonSenseSimilarities(sMod, sOtherMod)
                    featureVector = [pmiRatio] + wnSims + [dotV1V1o]
                    modFv.appendAltMod(sOtherMod, featureVector)
                    modFvCsv.writerow([sOtherMod] + featureVector)
                    numSiblingFiltered += 1
                # end of otherMod loop
                numSiblingToFreq[numSiblingFiltered] += 1
                pickle.dump(modFv, modFvPickleFile)
                # json.dump(modFv, modFvPickleFile)
                # end of instance loop
        # close files
        logging.info("%d phrases accepted", numInstAccepted)


class NccSubstDiagnosis(object):
    """Split off from NccSubstPrepare and based on object to reduce 
    loading time."""

    def __init__(self, flags):
        self.flags = flags
        self.phrasesPath = flags.phrases
        self.phrasesDir = os.path.dirname(os.path.realpath(self.phrasesPath))
        logging.info("NccSubstDiagnosis.__init__")

    def doDiagnosis(self, model=NccSubstModel()):
        """Loads pickled file of ModFv records and evaluates against gold labels.
        Saves to a dict with key = 'mod,head' and values = gold label, system label.
        """
        modFvPicklePath = os.path.join(self.flags.outdir, NccConst.modFvPickleName)
        logging.info("Loading %s", modFvPicklePath)
        xToYs = dict()  # mod,head --> ygold, ysys
        with open(modFvPicklePath, 'rb') as modFvPickleFile:
            while True:
                try:
                    modFv = pickle.load(modFvPickleFile)
                    xToYs[','.join((modFv.sMod, modFv.sHead))] = \
                        [modFv.y, model.prediction(modFv)]
                    if len(xToYs) % 1000 == 0:
                        logging.info("Loaded %d", len(xToYs))
                except EOFError:
                    break
        substPicklePath = os.path.join(self.phrasesDir, NccConst.substPickleName)
        with open(substPicklePath, 'wb') as substPickleFile:
            pickle.dump(xToYs, substPickleFile)
        nccEval = EvalRecallPrecisionF1()
        res = nccEval.doEvalDict(xToYs)
        logging.info("model=%s rho=%g f1=%g p100=%g ndcg2=%g",
                     str(model), res['rho'], res['maxf1'],
                     res['p100'], res['ndcg2'])

    def gridSearch(self):
        #     featureWeights = np.matrix('1; .1; .1; .1; .2')
        for l0 in [.8, .9, 1., 1.1, 1.2]:
            for l1 in [.1, .2, .3]:
                for l2 in [.1, .2, .3]:
                    for l3 in [.1, .2, .3]:
                        for l4 in [.1, .2, .3, .4, .5]:
                            for fb in [.0, .05, .1]:
                                for nss in [1, 1.5, 2., 2.5]:
                                    for nsb in [.5, 1., 2.]:
                                        model = NccSubstModel([l0, l1, l2, l3, l4], fb, nss, nsb)
                                        self.doDiagnosis(model)

    def printBadMistakes(self):
        """Sorts instances in decreasing order of |ygold - ysys|. Assumes
        both are in some range like [0,1]."""
        substPicklePath = os.path.join(self.phrasesDir, self.substPickleName)
        with open(substPicklePath, 'rb') as substPickleFile:
            ydiffs, phrases = [], []
            xToYs = pickle.load(substPickleFile)
            for k, y2 in xToYs.iteritems():
                ydiffs.append(abs(y2[0] - y2[1]))
                phrases.append(k)
        perm = np.array(ydiffs).argsort()[::-1][:len(ydiffs)]
        for ix in xrange(10):
            print ydiffs[perm[ix]], ', ', phrases[perm[ix]], xToYs[phrases[perm[ix]]]
        print>> sys.stderr, 'printBadMistakes {} {}'.format(len(phrases), len(ydiffs))


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True, type=str)
    parser.add_argument("--phrases", required=True, type=str)
    parser.add_argument("--ngramdir", required=True, type=str)
    parser.add_argument("--outdir", required=True, type=str)
    args = parser.parse_args()
    with tf.Session():
        psp = NccSubstPrepare(args)
        psp.saveModFv()
        psd = NccSubstDiagnosis(args)
        psd.doDiagnosis()
        # psd.gridSearch()
        # psd.printBadMistakes()
