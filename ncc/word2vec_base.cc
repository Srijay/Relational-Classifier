/**
 * Implementation of most of the methods in word2vec.h.
 */

#include <unistd.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <sstream>
#include <bitset>

#include "ncc.pb.h"
#include "word2vec.h"
#include "seqzfile.h"
#include "word2vec_loss.h"

namespace ncc {

  using std::cerr;
  using std::endl;		
  using std::vector;
  namespace fs = boost::filesystem;

  std::ostream cnull(0);

Word2vec &Word2vec::construct() {
  cerr << "Word2vec::conf = [[ " << conf.DebugString() << " ]]\n";

  mus.load(inpath + "/alphabet.pbs.gz");
  vocabSize = mus.vocabSize();
  std::size_t embedSize = vocabSize;
  embedSize *= conf.dim();
  cerr << "Word2vec::construct " << embedSize << endl;
  focusEmbed = new float[embedSize];
  contextEmbed = new float[embedSize];
  std::uniform_real_distribution<float> unif(-.5/conf.dim(), .5/conf.dim());
  for (int wid = 0; wid < mus.vocabSize(); ++wid) {
    float *const focusRow = getRow(focusEmbed, wid);
    float *const contextRow = getRow(contextEmbed, wid);
    for (int dim = 0; dim < conf.dim(); ++dim) {
      focusRow[dim] = unif(generator);
      contextRow[dim] = 0;
    }
  }
  radius = 1;
  return *this;
}

Word2vec::~Word2vec() {
  if (focusEmbed != nullptr) delete focusEmbed;
  if (contextEmbed != nullptr) delete contextEmbed;
}

Word2vec &Word2vec::checkInputSanity() {
  uint64_t numSentence = 0;
  auto corpusPath = inpath + "/corpus.pbs.gz";
  cerr << "Checking input corpus at " << corpusPath << endl;
  assert( std::ifstream(corpusPath).good() );


  SeqzFileReader corpusr(corpusPath);
  {
    string dteStr;
    DocTokEnt dte;
    for ( ; corpusr.get(&dteStr); ++numSentence) {
      assert ( dte.ParseFromString(dteStr) );
      for (const auto &wid : dte.tok_ids()) {
        CHECK_LE(-1, wid);
        CHECK_LT(wid, vocabSize);
      }

      const int ntoks = dte.tok_ids_size();
      const int nannots = dte.ent_ids_size();
      CHECK_EQ(nannots, dte.ent_begins_size());
      CHECK_EQ(nannots, dte.ent_ends_size());
      for (int ax = 0; ax < nannots; ++ax) {
        CHECK_LE(0, dte.ent_begins(ax));
        CHECK_LT(dte.ent_begins(ax), ntoks);
        CHECK_LE(0, dte.ent_ends(ax));
        CHECK_LT(dte.ent_ends(ax), ntoks);
        CHECK_LE(dte.ent_begins(ax), dte.ent_ends(ax));
        CHECK_LE(-1, dte.ent_ids(ax));
        CHECK_LT(dte.ent_ids(ax), vocabSize);
      }
      if (numSentence % 10000 == 0) {
        cerr << "Lines scanned " << numSentence << "\r";
      }
    }
    corpusr.close();
    cerr << endl;
  }
  return *this;
}


#define W2V_SWAP 1
#define W2V_RANDOM 0

/**
 * Simulates Mikolov code with our doc protos, mainly for regression alert.
 * Tests effects of swapping focus and context, and
 * using a deterministic hash for window and negative samples rather than
 * a pseudo random number generator.
 * A batch is one positive and a few negative pairs with the same focus word.
 */
void Word2vec::sampleCorpusMikolov() {
  Timer timer;
  const int window = conf.window();
  RecentLosses recentLosses;
  real *focusTemp = new real[conf.dim()];
  DocTokEnt dte;
  SamplerHash samplerHash;
#if W2V_RANDOM
  uint64_t seed = fixedSeed;
#endif

  const string &corpusPath = inpath + "/corpus.pbs.gz";
  assert( std::ifstream(corpusPath.c_str()).good() );
  SeqzFileReader corpusr(corpusPath);
  {
    string dteStr;
    DocTokEnt dte;
    int numSentence = 1;
    for ( ; corpusr.get(&dteStr); ++numSentence) {
      assert( dte.ParseFromString(dteStr) );
      const int sentenceSize = dte.tok_ids_size();
      avgSenLen.insert(sentenceSize);
      for (int focusPos = 0; focusPos < (int) sentenceSize; ++focusPos) {
        const int32 focusWid = dte.tok_ids(focusPos);
        if (focusWid <= 0) continue;
#if W2V_RANDOM
        const int bbb = SamplerHash::bump(&seed) % window;
#else
        const int bbb = samplerHash.hashToWindowSize(numSentence, sentenceSize, focusPos) % window;
#endif
        CHECK_LE(0, bbb);
        CHECK_LT(bbb, window);
        for (int aaa = bbb; aaa < window * 2 + 1 - bbb; ++aaa) {
          if (aaa == window) continue;
          const int contextPos = focusPos - window + aaa;
          if (contextPos < 0 || contextPos == focusPos || contextPos >= (int) sentenceSize) continue;
          const int32 contextWid = dte.tok_ids(contextPos);
          if (contextWid <= 0) continue;
          // contextWid == focusWid is allowed
          avgGap.insert(std::abs(focusPos-contextPos));
#if W2V_SWAP
          const int32 wida = contextWid;
          const int32 widb = focusWid;
#else
          const int32 wida = focusWid;
          const int32 widb = contextWid;
#endif
          beginSample(wida, widb, focusTemp);
          recentLosses.add_loss(putOneSample(wida, widb, 1, focusTemp));
          reportPair(numSentence, sentenceSize, focusPos, contextPos, wida, widb, 0, 0, 1);
          int actualNegsSampled = 0;
          for (int nx = 1; nx <= conf.neg(); ++nx) {
#if W2V_RANDOM
            const int64 ncx = SamplerHash::bump(&seed);
#else
            const int64 ncx = samplerHash.hashToNegativeWord(numSentence, sentenceSize, focusPos, contextPos, nx);
#endif
            int32 negContextWid = mus.sample(ncx, TokenKind::Token);
            if (negContextWid == 0) {
              negContextWid = 1 + ncx % (vocabSize-1);
            }
            if (negContextWid == focusWid) continue;
            // negContextWid == contextWid is allowed
            recentLosses.add_loss(putOneSample(wida, negContextWid, 0, focusTemp));
            reportPair(numSentence, sentenceSize, focusPos, contextPos, wida, negContextWid, nx, ncx % mus.numSlots(), 0);
            allNegWords.insert(negContextWid);
            ++actualNegsSampled;
          } // negative
          avgNeg.insert(actualNegsSampled);
          endSample(wida, widb, focusTemp);
        } // context
      } // focus
      if (timer.test()) {
        logProgress(recentLosses);
        saveEmbeddings();
      }
    } // sentence
    logProgress(recentLosses);
    saveEmbeddings();
    corpusr.close();
  }  // shards

  delete [] focusTemp;
}


/**
 * Mikolov style word pair sampler, but updates embeddings after every
 * word pair, instead of a batch with one positive and a few negative pairs.
 */
void Word2vec::sampleCorpusOnePair() {
  Timer timerLog, timerCheckpoint;
  const int32 window = conf.window();
  LogitDotUpdater<real> updater(conf.dim(), mus.vocabSize());
  RecentLosses recentLosses;
  SamplerHash samplerHash;
  uint64_t seed = fixedSeed;
  int numSentence = 0;
  const string &corpusPath = inpath + "/corpus.pbs.gz";
  assert( std::ifstream(corpusPath.c_str()).good() );
  {
    string dteStr;
    DocTokEnt dte;
    ncc::SeqzFileReader corpusr(corpusPath);
    for ( ; corpusr.get(&dteStr); ++numSentence) {
      const bool rc = dte.ParseFromString(dteStr); assert(rc);
      const int sentenceSize = dte.tok_ids_size();
      avgSenLen.insert(sentenceSize);
      for (int focusPos = 0; focusPos < (int) sentenceSize; ++focusPos) {
        const int32 focusWid = dte.tok_ids(focusPos);
        if (focusWid <= 0) continue;
        const int bbb = SamplerHash::bump(&seed) % window;
        CHECK_LE(0, bbb);
        CHECK_LT(bbb, window);
        for (int aaa = bbb; aaa < window * 2 + 1 - bbb; ++aaa) {
          if (aaa == window) continue;
          const int contextPos = focusPos - window + aaa;
          if (contextPos < 0 || contextPos == focusPos || contextPos >= (int) sentenceSize) continue;
          const int32 posContextWid = dte.tok_ids(contextPos);
          if (posContextWid <= 0) continue;
          // contextWid == focusWid is allowed
          avgGap.insert(std::abs(focusPos-contextPos));
          ++numPairs;
          recentLosses.add_loss(updater.trainPair(getRow(focusEmbed, focusWid),
                                                  getRow(contextEmbed, posContextWid),
                                                  1, conf.step()));
          ///recentLosses.add_loss(putOnePairSample(focusWid, posContextWid, 1, focusTemp));
          int actualNegsSampled = 0;
          for (int nx = 1; nx <= conf.neg(); ++nx) {
            const int64 ncx = SamplerHash::bump(&seed);
            const int32 negContextWid = mus.sample(ncx, TokenKind::Token);
//            if (negContextWid == 0) {
//              negContextWid = 1 + ncx % (vocabSize-1);
//            }
            if (negContextWid == focusWid) continue;
            // negContextWid == contextWid is allowed
            ++numPairs;
            recentLosses.add_loss(updater.trainPair(getRow(focusEmbed, focusWid),
                                                    getRow(contextEmbed, negContextWid),
                                                    0, conf.step()));
            allNegWords.insert(negContextWid);
            ++actualNegsSampled;
          } // negative
          avgNeg.insert(actualNegsSampled);
        } // context sweep
      } // focus sweep
      if (timerLog.test(base::Seconds(30))) {
        logProgress(recentLosses);
      }
      if (timerCheckpoint.test()) {
        saveEmbeddings();
      }
    } // lines
    corpusr.close();
  } // shard
  logProgress(recentLosses);
  saveEmbeddings();
}


void Word2vec::produceTrainingPairs(ProducerConsumerQueue<TrainWork> *pcq, bool *pDone) {
  Timer timer;
  const int32 window = conf.window();
  const string &corpusPath = inpath + "/corpus.pbs.gz";
  assert( std::ifstream(corpusPath.c_str()).good() );
  {
    DocTokEnt dte;
    string dteStr;
    SeqzFileReader corpusr(corpusPath);
    uint64_t numPairs = 0;
    while (corpusr.get(&dteStr)) {
      const bool rc = dte.ParseFromString(dteStr); assert(rc);
      avgSenLen.insert(dte.tok_ids_size());
      for (int focusPos = 0; focusPos < dte.tok_ids_size(); ++focusPos) {
        const int32 &focusWid = dte.tok_ids(focusPos);
        CHECK_LT(focusWid, vocabSize);
        if (focusWid < 0) continue;
        auto *work = new TrainWork;
        work->set_focus_tid(focusWid);
        work->set_focus_kind(TokenKind::Token);
        for (int numContext=0, contextPos=focusPos-1;
            contextPos >= 0 && numContext < window; --contextPos) {
          const int32 &contextWid = dte.tok_ids(contextPos);
          if (contextWid < 0) continue;
          // contextWid == focusWid is allowed
          CHECK_LT(contextWid, vocabSize);
          avgGap.insert(std::abs(focusPos-contextPos));
          work->add_context_tids(contextWid);
          work->add_context_kinds(TokenKind::Token);
          ++numPairs;
          ++numContext;
        }  // left context
        for (int numContext=0, contextPos = focusPos+1;
            contextPos < dte.tok_ids_size() && numContext < window; ++contextPos) {
          const int32 &contextWid = dte.tok_ids(contextPos);
          if (contextWid < 0) continue;
          // contextWid == focusWid is allowed
          CHECK_LT(contextWid, vocabSize);
          avgGap.insert(std::abs(focusPos-contextPos));
          work->add_context_tids(contextWid);
          work->add_context_kinds(TokenKind::Token);
          ++numPairs;
          ++numContext;
        }  // right context
        pcq->push(*work);
	delete work;
      } // focus
      if (timer.test()) {
	cerr << format("Produced %lu positive pairs\n", numPairs);
        saveEmbeddings();
      }
    } // line
    corpusr.close();
  }
  *pDone = true;
  cerr << format("Produced %lu positive pairs\n", numPairs);
}

//@Deprecated
void Word2vec::consumeOneTrainingRecord(TrainWork *work2,
                                        RecentLosses *recentLosses,
                                        real *focusTemp,
                                        SamplerHash *hash,
                                        uint64_t *seed,
                                        uint64_t *numPosPairs,
                                        uint64_t *numNegPairs)
{
  for (int cx = 0; cx < work2->context_tids_size(); ++cx) {
    const int32 &focusTid = work2->focus_tid(),
        &contextTid = work2->context_tids(cx);
    ++(*numPosPairs);
    beginSample(focusTid, contextTid, focusTemp);
    recentLosses->add_loss( putOneSample(focusTid, contextTid, 1, focusTemp) );
    for (int nx = 1; nx <= conf.neg(); ++nx) {
      const int64 ncx = SamplerHash::bump(seed);
      const int32 negContextWid = mus.sample(ncx, TokenKind::Token);
      if (negContextWid == 0) continue;
      if (negContextWid == focusTid) continue;
      // negContextWid == contextWid is allowed
      ++(*numNegPairs);
      recentLosses->add_loss( putOneSample(focusTid, negContextWid, 0, focusTemp) );
      // allNegWords.insert(negContextWid);
    } // negative
    endSample(focusTid, contextTid, focusTemp);
  } // context
}


void Word2vec::consumeTrainingPairs(int threadId,
                                    ProducerConsumerQueue<TrainWork> *pcq,
                                    bool *pDone)
{
  Timer timer;
  RecentLosses recentLosses;
  LogitDotUpdater<real>
  // LogitNdist2Updater<real>
  updater(conf.dim(), mus.vocabSize());
  SamplerHash samplerHash;
  uint64_t seed = threadId + fixedSeed;
  uint64_t numPosPairs = 0, numNegPairs = 0;
  for (;;) {
    TrainWork work;
    work.Clear();
    const bool got = pcq->pop(&work, std::chrono::milliseconds(1000));
    if (!got and *pDone) {
      cerr << format("Thread %d finished\n",  threadId);
      break;
    }
    auto *work2 = (TrainWork *) &work;
    checkWorkSanity(work2);
    //------ work -------
    const int32 &focusTid = work2->focus_tid();
    for (const int32 &posContextTid : work2->context_tids()) {
      ++(numPosPairs);
      for (int nx = 0; nx <= conf.neg(); ++nx) {
	int32 contextTid = posContextTid;
	if (nx > 0) {
	  const int64 ncx = SamplerHash::bump(&seed);
	  const int32 negContextTid = mus.sample(ncx, TokenKind::Token);
	  if (negContextTid == 0) continue;
	  if (negContextTid == focusTid) continue;
	  // negContextWid == contextWid is allowed
	  ++(numNegPairs);
	  contextTid = negContextTid;
	}  // negative
	const auto label = nx == 0? 1. : 0.;
	recentLosses.add_loss(updater.trainPair(getRow(focusEmbed, focusTid),
						getRow(contextEmbed, contextTid),
						//&radius,
						label, conf.step()) );
      }
    } // pos-context
    //------ work -------
    if (timer.test()) {
      cerr << format("T#%d consumed %lu pos %lu neg pairs; recent loss = %g\n",
		     threadId, numPosPairs, numNegPairs, recentLosses.average_loss());
    }
  }  // forever
}

void Word2vec::checkWorkSanity(TrainWork *work) {
  CHECK_LE(0, work->focus_tid());
  CHECK_LT(work->focus_tid(), vocabSize);
  const auto &ncon = work->context_tids_size();
  CHECK_EQ(ncon, work->context_kinds_size());
  for (int cx = 0; cx < ncon; ++cx) {
    CHECK_LE(0, work->context_tids(cx));
    CHECK_LT(work->context_tids(cx), vocabSize);
    CHECK_EQ(work->context_kinds(cx), TokenKind::Token);
  }
}

void Word2vec::sampleCorpus(const int numThreads) {
  ProducerConsumerQueue<TrainWork> pcq;
  bool pDone = false;
  auto *pool = new ncc::ThreadPool(numThreads);
  {
    std::function<void(void)> clo =
      std::bind(&Word2vec::produceTrainingPairs, this, &pcq, &pDone);
    pool->Enqueue(clo);
    //pool->Add(NewCallback(this, &Word2vec::produceTrainingPairs, &pcq, &pDone));
  }
  for (int cx = 0; cx < numThreads; ++cx) {
    std::function<void(void)> clo =
      std::bind(&Word2vec::consumeTrainingPairs, this, cx, &pcq, &pDone);
    pool->Enqueue(clo);
    //pool->Add(NewCallback(this, &Word2vec::consumeTrainingPairs, cx, &pcq, &pDone));
  }
  while (!pDone) sleep(1);
  cerr << "Deleting pool\n";
  delete pool;
  saveEmbeddings();
}

void Word2vec::produceTrainingPairsWithAnnotations(ProducerConsumerQueue<TrainWork> *pcq,
                                                   bool *pDone) {
  Timer timer;
  const auto window = conf.window();
  const string &corpusPath = inpath + "/corpus.pbs.gz";
  assert( std::ifstream(corpusPath.c_str()).good() );
  {
    cerr << "Scanning " << corpusPath << endl;
    SeqzFileReader corpusr(corpusPath);
    DocTokEnt dte;
    string dteStr;
    int64 numLines = 0;
    while (corpusr.get(&dteStr)) {
      ++numLines;
      if (timer.test()) {
	std::cerr <<
	  format("Scanned %lld lines, produced %llu positive pairs\n",
		 numLines, numPairs);
      }
      const bool rc = dte.ParseFromString(dteStr); assert(rc);
      avgSenLen.insert(dte.tok_ids_size());
      // focus = text  --- TODO(soumenc) refactor
      avgSenLen.insert(dte.tok_ids_size());
      for (int focusPos = 0; focusPos < dte.tok_ids_size(); ++focusPos) {
        const auto &focusWid = dte.tok_ids(focusPos);
        if (focusWid < 0) continue;
        CHECK_LT(focusWid, vocabSize);
        auto *work = new TrainWork;
        work->set_focus_tid(focusWid);
        work->set_focus_kind(TokenKind::Token);
        for (int numContext=0, contextPos=focusPos-1;
            contextPos >= 0 && numContext < window; --contextPos) {
          const auto &contextWid = dte.tok_ids(contextPos);
          if (contextWid < 0) continue;
          // contextWid == focusWid is allowed
          CHECK_LT(contextWid, vocabSize);
          avgGap.insert(std::abs(focusPos-contextPos));
          work->add_context_tids(contextWid);
          work->add_context_kinds(TokenKind::Token);
          ++numPairs;
          ++numContext;
        }  // left context
        for (int numContext=0, contextPos = focusPos+1;
            contextPos < dte.tok_ids_size() && numContext < window; ++contextPos) {
          const auto &contextWid = dte.tok_ids(contextPos);
          if (contextWid < 0) continue;
          // contextWid == focusWid is allowed
          CHECK_LT(contextWid, vocabSize);
          avgGap.insert(std::abs(focusPos-contextPos));
          work->add_context_tids(contextWid);
          work->add_context_kinds(TokenKind::Token);
          ++numPairs;
          ++numContext;
        }  // right context
        pcq->push(*work);
	delete work; // pushed by copy so we can delete
      } // focus

      // focus = annotation
      CHECK_EQ(dte.ent_begins_size(), dte.ent_ends_size());
      CHECK_EQ(dte.ent_begins_size(), dte.ent_ids_size());
      for (int ex = 0; ex < dte.ent_begins_size(); ++ex) {
        const auto &focusWid = dte.ent_ids(ex);
        if (focusWid < 0) continue;
        CHECK_LT(focusWid, vocabSize);
        auto *work = new TrainWork;
        work->set_focus_tid(focusWid);
        work->set_focus_kind(TokenKind::Entity);
        const auto &ebegin = dte.ent_begins(ex), &eend = dte.ent_ends(ex);
        for (int numContext=0, contextPos=ebegin-1;
            contextPos >= 0 && numContext < window; --contextPos) {
          const auto &contextWid = dte.tok_ids(contextPos);
          if (contextWid < 0) continue;
          // contextWid == focusWid is allowed
          CHECK_LT(contextWid, vocabSize);
          avgGap.insert(std::abs(ebegin-contextPos));
          work->add_context_tids(contextWid);
          work->add_context_kinds(TokenKind::Token);
          ++numPairs;
          ++numContext;
        }  // left context
        for (int numContext=0, contextPos = eend+1;
            contextPos < dte.tok_ids_size() && numContext < window; ++contextPos) {
          const auto &contextWid = dte.tok_ids(contextPos);
          if (contextWid < 0) continue;
          // contextWid == focusWid is allowed
          CHECK_LT(contextWid, vocabSize);
          avgGap.insert(std::abs(eend-contextPos));
          work->add_context_tids(contextWid);
          work->add_context_kinds(TokenKind::Token);
          ++numPairs;
          ++numContext;
        }  // right context
        pcq->push(*work);
	delete work; // pushed by copy so we can delete
      }   // focus
    } // lines
    corpusr.close();
    cerr << "Closed " << corpusPath << endl;
    saveEmbeddings();
  }  // for each shard
  *pDone = true;
  cerr << format("Producer completed %lu positive pairs\n", numPairs);
}

void Word2vec::sampleCorpusWithAnnotations(const int numThreads) {
  ProducerConsumerQueue<TrainWork> pcq;
  bool pDone = false;
  auto *pool = new ncc::ThreadPool(numThreads);
  {
    std::function<void(void)> clo =
      std::bind(&Word2vec::produceTrainingPairsWithAnnotations, this, &pcq, &pDone);
    pool->Enqueue(clo);
    //pool->Add(NewCallback(this, &Word2vec::produceTrainingPairsWithAnnotations, &pcq, &pDone));
  }  
  for (int cx = 0; cx < numThreads; ++cx) {
    std::function<void(void)> clo =
      std::bind(&Word2vec::consumeTrainingPairs, this, cx, &pcq, &pDone);
    pool->Enqueue(clo);
    //pool->Add(NewCallback(this, &Word2vec::consumeTrainingPairs, cx, &pcq, &pDone));
  }
  while (!pDone) sleep(1);
  cerr << "Deleting pool\n";
  delete pool;
  saveEmbeddings();
}

void Word2vec::nearest(const std::vector<WordId> &queryWids, size_t topk,
                       std::vector<ResponseHeap> *responses)
{
  responses->clear();
  responses->resize(queryWids.size());
  for (size_t qx = 0; qx < queryWids.size(); ++qx) {
    const auto queryWid = queryWids[qx];
    const real *queryRow = getRow(focusEmbed, queryWid);
    auto &queryResponseHeap = (*responses)[qx];
    while (!queryResponseHeap.empty()) queryResponseHeap.pop();
    for (WordId nearWid = 0; nearWid < vocabSize; ++nearWid) {
      if (nearWid == queryWid) continue;
      const real *nearRow = getRow(focusEmbed, nearWid);
      real dot=0, norm2=0;
      for (int dim = 0; dim < conf.dim(); ++dim) {
        dot += queryRow[dim] * nearRow[dim];
        norm2 += nearRow[dim] * nearRow[dim];
      }
      const auto norm = sqrt(norm2);
      queryResponseHeap.push(std::make_pair(-dot/norm, nearWid));
      if (queryResponseHeap.size() > topk) {
        queryResponseHeap.pop();
      }
    }  // for each potentially near word
  } // for each query word
}

void Word2vec::logProgress(RecentLosses &recent) {
  // "%c[2K%c" 27 13
  cerr << format("RecentLoss=%5.3f AvgSenLen=%6.3g AvgGap=%6.3g AvgNeg=%6.3g NegWords=%8ld",
		 recent.average_loss(), avgSenLen.get(), avgGap.get(),
		 avgNeg.get(), allNegWords.size());
  // Print near neighbors of a few probe words.
  cerr << "\tpairs=" << numPairs << std::endl;
  std::vector<WordId> queryWids;
  for (const auto &queryWord : queryWords) {
    const auto queryWid = mus.tokenToTid(TokenKind::Token, queryWord);
    if (queryWid < 0) continue;
    queryWids.push_back(queryWid);
  }
  const int topk = 5;
  std::vector<ResponseHeap> responses;
  nearest(queryWids, topk, &responses);
  for (size_t qx = 0; qx < queryWids.size(); ++qx) {
    const auto queryWid = queryWids[qx];
    TokenKind queryKind = TokenKind::Null;
    string queryToken;
    CHECK( mus.tidToToken(queryWid, &queryKind, &queryToken) );
    cerr << "\t" << queryToken << ": ";
    auto &queryResponseHeap = responses[qx];
    while (!queryResponseHeap.empty()) {
      const auto &response = queryResponseHeap.top();
      TokenKind respKind = TokenKind::Null;
      string respToken;
      CHECK( mus.tidToToken(response.second, &respKind, &respToken));
      cerr << respToken << "," << -response.first << " ";
      queryResponseHeap.pop();
    }
    cerr << std::endl;
  }
  numBatches = numPairs / pairsPerBatch;
}

void Word2vec::normalizeFocusEmbeddings() {
  cerr << "Normalizing focus vectors.\n";
  for (int row = 0; row < mus.vocabSize(); ++row) {
    auto *focusRow = getRow(focusEmbed, row);
    real l2 = 0;
    for (int col = 0; col < conf.dim(); ++col) {
      l2 += focusRow[col] * focusRow[col];
    }
    if (l2 < std::numeric_limits<real>::min()) return;
    l2 = std::sqrt(l2);
    for (int col = 0; col < conf.dim(); ++col) {
      focusRow[col] /= l2;
    }
  }
  cerr << "Finished normalizing focus vectors.\n";
}

void Word2vec::saveEmbeddingsMikolov() {
  cerr << "WARNING: saveEmbeddingsMikolov disabled.\n";
  return;
  const string &embedPath = outdir + "/embed.w2v";
  FILE *fo = fopen(embedPath.c_str(), "wb");
  fprintf(fo, "%lld %lld\n", (long long) vocabSize, (long long) conf.dim());
  for (std::size_t a = 0; a < vocabSize; a++) {
    TokenKind kind = TokenKind::Null;
    string token;
    CHECK( mus.tidToToken(a, &kind, &token) );
    // TODO(soumenc) Save TokenKind too.
    fprintf(fo, "%s ", token.c_str());
    const std::size_t _dim = conf.dim();
    for (std::size_t b = 0; b < conf.dim(); b++) {
      fwrite(focusEmbed + a * _dim + b, sizeof(real), 1, fo);
    }
    fprintf(fo, "\n");
  }
  fclose(fo);
  cerr << "Wrote focusEmbed to " << embedPath << endl;
}

void Word2vec::saveEmbeddingsNumpy() {
  const string &npMatrixPath = outdir + "/embed.numpy";
  std::size_t numBytes = sizeof(real);
  numBytes *= vocabSize;
  numBytes *= conf.dim();
  cerr << "Saving " << numBytes << " bytes of embeddings to "
       << npMatrixPath << endl;
  ofstream npMatrixOfs(npMatrixPath, ios::binary);
  npMatrixOfs.write((char*)focusEmbed, numBytes);
  npMatrixOfs.close();
  const string &npMetaPath = outdir + "/embed.meta";
  ofstream npMetaOfs(npMetaPath);
  npMetaOfs << vocabSize << endl << conf.dim() << endl;
  npMetaOfs.close();
}

void Word2vec::loadEmbeddingsNumpy() {
  const string &npMetaPath = outdir + "/embed.meta";
  ifstream npMetaIfs(npMetaPath);
  int32 loadedVocabSize = 0, loadedDim = 0;
  npMetaIfs >> loadedVocabSize >> loadedDim;
  npMetaIfs.close();
  const auto sameShape = (vocabSize == loadedVocabSize
			  and conf.dim() == loadedDim);
  if (!sameShape) {
    cerr << "FATAL: trying to load pretrained embeddings with wrong shape\n";
  }
  assert(sameShape);
  const string &npMatrixPath = outdir + "/embed.numpy";
  std::size_t numBytes = sizeof(real);
  numBytes *= vocabSize;
  numBytes *= conf.dim();
  cerr << "Loading " << numBytes << "bytes of embeddings from "
       << npMatrixPath << endl;
  ifstream npMatrixIfs(npMatrixPath, ios::binary);
  npMatrixIfs.read((char*)focusEmbed, numBytes);
  assert(npMatrixIfs);
  npMatrixIfs.close();
}

// Cannot Word2vec::saveEmbeddingsSstable()
// Cannot Word2vec::loadEmbeddings[Sstable]()

}  // namespace ncc

