#ifndef _WORD2VEC_H_
#define _WORD2VEC_H_ 1

#include <stddef.h>
#include <time.h>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <map>
#include <queue>
#include <random>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdarg>
#include <ext/hash_set>
#include <ext/hash_map>
#include <utility>
#include <vector>
#include <cassert>
#include <cstdint>
#include <openssl/sha.h>

#include "ncc.pb.h"
#include "word2vec_loss.h"
#include "pcqueue.h"
#include "threadpool.h"

/**
 * These may not actually log to cerr, but helps compile
 * code like CHECK(foo) << "message if fails";
 */
namespace ncc {
  extern std::ostream cnull;
}
#define CHECK(b) assert((b)), (b? ncc::cnull : std::cerr)
#define CHECK_LE(a, b) assert((a) <= (b)), ((a)<=(b)? ncc::cnull : std::cerr)
#define CHECK_LT(a, b) assert((a) <  (b)), ((a)< (b)? ncc::cnull : std::cerr)
#define CHECK_GT(a, b) assert((a) >  (b)), ((a)> (b)? ncc::cnull : std::cerr)
#define CHECK_EQ(a, b) assert((a) == (b)), ((a)==(b)? ncc::cnull : std::cerr)

typedef uint64_t uint64;
typedef int64_t int64;
typedef int32_t int32;
typedef float real;

using std::string;
using std::vector;
using __gnu_cxx::hash_map;
using __gnu_cxx::hash_set;
using __gnu_cxx::hash_multimap;
using std::cerr;
using std::cout;
using std::endl;
using std::ofstream;

namespace __gnu_cxx {
/**
        Explicit template specialization of hash of a string class,
        which just uses the internal char* representation as a wrapper.
 */
template <>
struct hash<std::string> {
  size_t operator() (const std::string &x) const {
    return hash<const char*>()(x.c_str());
    // hash<const char*> already exists
  }
};
}  // namespace __gnu_cxx

struct base {
  typedef unsigned char uchar;
  typedef time_t Time;
  typedef double Duration;
  
  static Time Now() {
    time_t clock;
    time(&clock);
    return clock;
  }
  
  static Duration Seconds(double s) { return s; }

  static Duration Diff(const Time &a, const Time &b) {
    return difftime(a, b);
  }

  static int64_t Fingerprint2011(const uchar *buf, int len) {
    char out[SHA256_DIGEST_LENGTH];
    SHA256(buf, len, (uchar*) out);
    int64_t h0, h1, h2, h3;
    strncpy((char*) &h0, out, sizeof(h0));
    strncpy((char*) &h1, out+sizeof(h0), sizeof(h1));
    strncpy((char*) &h2, out+sizeof(h0)+sizeof(h1), sizeof(h2));
    strncpy((char*) &h3, out+sizeof(h0)+sizeof(h1)+sizeof(h2), sizeof(h3));
    return h0 ^ h1 ^ h2 ^ h3;
  }

  static int64_t FingerprintCat2011(int64_t a, int64_t b) {
    const int64_t kMul1 = 0xc6a4a7935bd1e995ULL;
    const int64_t kMul2 = 0x228876a7198b743ULL;
    int64_t c = a * kMul1 + b * kMul2;
    return c + (~c >> 47);
  }
};  // struct base

namespace ncc {

//missing string printf
//this is safe and convenient but not exactly efficient
inline std::string format(const char* fmt, ...){
    int size = 512;
    char* buffer = 0;
    buffer = new char[size];
    va_list vl;
    va_start(vl,fmt);
    int nsize = vsnprintf(buffer,size,fmt,vl);
    if(size<=nsize){//fail delete buffer and try again
        delete buffer; buffer = 0;
        buffer = new char[nsize+1];//+1 for /0
        nsize = vsnprintf(buffer,size,fmt,vl);
    }
    std::string ret(buffer);
    va_end(vl);
    delete buffer;
    return ret;
}

class Timer {
 protected:
  base::Time last = base::Now();

 public:
  bool test(const base::Duration &since = base::Seconds(120)) {
    base::Time now = base::Now();
    //std::cout << base::Diff(now, last) << ' ' << since << std::endl;
    if (base::Diff(now, last) > since) {
      last = now;
      return true;
    }
    return false;
  }
};

class Average {
 protected:
  double sum=0, count=0;
 public:
  void insert(double val) {
    sum += val;
    ++count;
  }
  double get() const {
    return sum / count;
  }
  std::string toString() const {
    std::ostringstream stringStream;
    stringStream << sum << '/' << count << '=' << get();
    return stringStream.str();
  }
};


class RecentLosses : public std::deque<double> {
protected:
  const unsigned int max_losses = 1000000;
  double sum = 0;

public:
  void add_loss(double aloss) {
    push_back(aloss);
    sum += aloss;
    if (size() >= max_losses) {
      sum -= front();
      pop_front();
    }
  }

  double average_loss() {
    if (size() == 0) return 0;
    return sum / size();
  }
};


/**
 * Derandomizes choice of window and negative context words.
 */
class SamplerHash {
 protected:
  int debugLevel = 0;

  uint64 fp(int a) const {
    return base::Fingerprint2011((base::uchar*) &a, sizeof(a));
  }

 public:
  // Can afford this to be a little slow.
  uint64 hashToWindowSize(int sentence_number,
                          int sentence_size,
                          int context_position) const
  {
    uint64 ans3 = fp(sentence_number);
    ans3 = base::FingerprintCat2011(ans3, fp(sentence_size));
    ans3 = base::FingerprintCat2011(ans3, fp(context_position));
    return ans3;
  }

  // Can afford this to be a little slow.
  uint64 hashToNegativeWord(int sentence_number,
                            int sentence_size,
                            int context_position,
                            int focus_position,
                            int negative_trial) const
  {
    uint64 ans3 = fp(sentence_number);
    ans3 = base::FingerprintCat2011(ans3, fp(sentence_size));
    ans3 = base::FingerprintCat2011(ans3, fp(context_position));
    ans3 = base::FingerprintCat2011(ans3, fp(focus_position));
    ans3 = base::FingerprintCat2011(ans3, fp(negative_trial));
    return ans3;
  }

  static uint64_t bump(uint64_t *prnd) {
    *prnd = *prnd * (unsigned long long)25214903917 + 11;
    return *prnd;
  }
};


/**
 * Manages vocabulary and unigram sampling.
 * Also provides entity <--> type mappings.
 */
class MikolovUnigramSampler {
 public:
  /**
   * Upon construction, this object is empty and unusable.
   * This method fills in this object from a VocabWord recordio file.
   * Slots are also read from this recordio file.
   * Kinds not populated cannot be filled in later.
   * Kinds must occupy adjacent token IDs.
   */
  void load(const string &vocabRioPath);
  /**
   * Wipe out any previous slot info and re-compute based on probabilities.
   */
  void redoSlots(int32 numSlots, double unipower);
  /**
   * Save back to recordio.
   */
  void save(const string &vocabRioPath);
  /**
   * Save to readable file.
   */
  void saveReadable(const string &uname);
  void saveSlots(const string &uname);
  /**
   * Common access methods.
   */
  // Returns -1 if not found.
  int tokenToTid(ncc::TokenKind kind, const string &token) const;
  // Returns false if there is trouble.
  bool tidToToken(int tid, ncc::TokenKind *kind, string *token) const;
  // Returns false if there is trouble.
  bool tidToSlots(int tid, int *slots) const;
  int vocabSize() const { return vocab.size(); }
  int numSlots() const { return wordPicker.size(); }

  /**
   * Sample using given random number generator.
   * If kind == Null, sample from all token kinds, otherwise restrict to kind.
   */
//  int32 sample(std::default_random_engine &rng, ncc::TokenKind kind=ncc::TokenKind::Null);

  /**
   * Sample using given slot integer to deference slot array after
   * range restrict.  (This is for deterministic debugging.)
   * If kind == Null, sample from all token kinds, otherwise restrict to kind.
   */
  int32 sample(uint64 rn, ncc::TokenKind kind=ncc::TokenKind::Null);

  const VocabWord &tidToVocab(const int tid) const {
    return vocab[tid];
  }

 protected:
  vector<VocabWord> vocab;  // wastes RAM but simpler
  const char kindTokenSep = '_';
  hash_map<string, int32> tokenToTidMap;  // key = "kind_word"
  std::map<ncc::TokenKind, int> kindToSlotBegin;
  std::map<ncc::TokenKind, int> kindToSlotEnd;
  vector<int> wordPicker;
  std::uniform_int_distribution<> iunif;
  void installSlots();
};

/**
 * Utility methods for phrase string processing.
 */
class PhraseBase {
 public:
  //  typedef StreamTrie<int, -1> PhraseTrie;
  //  typedef PhraseTrie::MatchedPhrase<vector<string>::iterator> PhraseMatch;

  /* static void split(const string &phrase, vector<string> *words) { */
  /*   SplitCSVLineWithDelimiterForStrings(phrase, '^', words); */
  /*   words->erase(std::remove_if(words->begin(), words->end(), */
  /*                               [](const string &s) { return s.empty(); }), */
  /*                words->end()); */
  /* } */

  static void split(const string &phrase, vector<string> *words) {
    std::stringstream ss(phrase);
    words->clear();
    string item;
    while (getline(ss, item, '^')) {
      words->push_back(item);
    }
    //SplitStructuredLine(phrase, '^', "", words);
    words->erase(std::remove_if(words->begin(), words->end(),
                                [](string &s) { return s.empty(); }),
                 words->end());
  }

  template <typename C> static void join(const C &coll, char delim, string *out) {
    out->clear();
    for (const auto &elem : coll) {
      out->append(elem);
      out->append(1, delim);
    }
  }

  static string toAnotherKey(const string &phrase, const char sep) {
    vector<string> words;
    split(phrase, &words);
    string out;
    join(words, sep, &out);
    return out;
    //return strings::Join(words.begin(), words.end(), sep);
  }

  /**
   * Input is caret-separated words, with trailing caret.
   * Output is underscore-separated without trailing underscore.
   */
  static string toBrainKey(const string &phrase) {
    return toAnotherKey(phrase, '_');
  }

  /**
   * Input is caret-separated words, with trailing caret.
   * Output is space-separated, without trailing space.
   */
  static string toNgramKey(const string &phrase) {
    return toAnotherKey(phrase, ' ');
  }

  static string join(const vector<string> &words) {
    string out;
    join(words, '^', &out);
    return out;
    //return strings::Join(words.begin(), words.end(), "^") + '^';
  }

  //  static void join(const PhraseMatch &match, vector<string> *words) {
  //    words->clear();
  //    for (auto mx = match.from; mx != match.to; ++mx) {
  //      words->push_back(*mx);
  //    }
  //  }

  //  static string joinMatch(const PhraseMatch &match) {
  //    vector<string> words;
  //    join(match, &words);
  //    return join(words);
  //  }

};

/**
 * Enhanced Word2vec that processes span annotations as additional words.
 */
class Word2vec : public PhraseBase {
 protected:
  const Config &conf;
  typedef float real;
  typedef int WordId;
  typedef std::pair<real, WordId> Response;
  typedef std::priority_queue<Response> ResponseHeap;
  const uint64_t fixedSeed = 1447553299L;

  string inpath, outdir;
  int vocabSize=-1;
  MikolovUnigramSampler mus;
  std::default_random_engine generator;

  real *focusEmbed=nullptr, *contextEmbed=nullptr;
  real radius = 1;  // in case of distance-based loss
  Average avgSenLen, avgGap, avgNeg;
  hash_set<int> allNegWords;
  uint64_t numPairs = 0, numBatches = 0, pairsPerBatch = 10000000;
  std::vector<std::string> queryWords {
    "april", "china", "gallon", "germany", "iron", "january",
    "monday", "steel", "wheat" };

  real *const getRow(real *const mat, std::size_t row) const {
    std::size_t ofs = row;
    ofs *= conf.dim();
    return mat + ofs;
  }

  real getLoss(real label, real dot) {
    const double loss =  (1. - label) * dot + ncc::LogitUtils<real>::log1pexp(-dot);
    return loss;
  }

  // Looks up current model and returns near neighbors for each query word ID.
  void nearest(const std::vector<WordId> &queryWids, size_t topk,
               std::vector<ResponseHeap> *responses);
  void logProgress(RecentLosses &recent);

  void consumeOneTrainingRecord(TrainWork *work2,
                                RecentLosses *recent,
                                real *focusTemp,
                                SamplerHash *hash,
                                uint64_t *seed,
                                uint64_t *numPosPairs,
                                uint64_t *numNegPairs);

  void op_consumeOneTrainingRecord(TrainWork *work2,
                                RecentLosses *recent,
                                SamplerHash *hash,
                                uint64_t *seed,
                                uint64_t *numPosPairs,
                                uint64_t *numNegPairs);

  /**
   * If trained with loss based on dot product of word vector pairs, those
   * vectors are usually scaled to unit length.  But if trained for
   * distance-based loss, scaling will generally mess up the vectors.
   */
  void normalizeFocusEmbeddings();

 public:
  Word2vec(const Config &conf_) : conf(conf_) { }

  virtual ~Word2vec();

  Word2vec &withDirs(const string &indir_, const string &outdir_) {
    inpath = indir_;
    outdir = outdir_;
    return *this;
  }

  // sentence_count, sentence_length, sentence_position, c, last_word, target, label

  static void reportPair(int nSen, int senLen, int focPos, int ctxPos, int focusWid, int contextWid, int negx, uint64 hashv, float label) {
//     printf("PAIR%12d%5d%5d%5d%12d%12d%5d%24llu%4g\n", nSen, senLen, focPos, ctxPos, focusWid, contextWid, negx, hashv, label);
  }

  // @Deprecated
  void beginSample(int focusWid, int contextWid, real *focusTemp) {
    for (int dim = 0; dim < conf.dim(); ++dim) {
      focusTemp[dim] = 0;
    }
  }

  // @Deprecated
  real putOneSample(int focusWid, int contextWid, real label, real *focusTemp) {
    //cerr << label << ": " << focusWid << "," << contextWid << endl;
    CHECK_LE(0, focusWid) << "Bad focusWid=" << focusWid;
    CHECK_LT(focusWid, vocabSize) << "Bad focusWid=" << focusWid;
    CHECK_LE(0, contextWid) << "Bad contextWid=" << contextWid;
    CHECK_LT(contextWid, vocabSize) << "Bad contextWid=" << contextWid;
    const real *const focusVec = getRow(focusEmbed, focusWid);
    real *const contextVec = getRow(contextEmbed, contextWid);
    const real adot = ncc::LogitUtils<real>::dot(focusVec, contextVec, conf.dim());
    const real prediction = ncc::LogitUtils<real>::logit(adot);
    const real gradPart = (label - prediction) * conf.step();
    for (int dim = 0; dim < conf.dim(); ++dim) {
      focusTemp[dim] += gradPart * contextVec[dim];
    }
    for (int dim = 0; dim < conf.dim(); ++dim) {
      contextVec[dim] += gradPart * focusVec[dim];
    }
    ++numPairs;
    const real loss = getLoss(label, adot);
    if (conf.debug() > 10) {
      ncc::TokenKind focusKind = ncc::TokenKind::Null, contextKind = ncc::TokenKind::Null;
      string focusToken, contextToken;
      CHECK( mus.tidToToken(focusWid, &focusKind, &focusToken) );
      CHECK( mus.tidToToken(contextWid, &contextKind, &contextToken) );
      fprintf(stdout, "SAMPLE\t%15s:%d\t%15s:%d\t%g\t%8.2g\n",
              focusToken.c_str(), focusWid, contextToken.c_str(), contextWid,
              label, loss);
    }
    return loss;
  }

  // @Deprecated
  void endSample(int focusWid, int contextWid, real *focusTemp) {
    real *const focusVec = getRow(focusEmbed, focusWid);
    for (int dim = 0; dim < conf.dim(); ++dim) {
      focusVec[dim] += focusTemp[dim];
    }
  }

  real putOnePairSample(int wida, int widb, real label, real *focusTemp) {
    beginSample(wida, widb, focusTemp);
    auto ans = putOneSample(wida, widb, label, focusTemp);
    endSample(wida, widb, focusTemp);
    return ans;
  }

  // Allocate and initialize arrays etc.
  virtual Word2vec &construct();

  // Check vocabulary vs. corpus word IDs.
  virtual Word2vec &checkInputSanity();

  void checkWorkSanity(TrainWork *);

  // Emulates classic Mikolov code with most of its magic.
  // Pays no attention to annotation spans.
  void sampleCorpusMikolov();

  // Train from one pair at a time.
  // Pays no attention to annotation spans.
  void sampleCorpusOnePair();

  // Scan corpus and generate training pairs.
  // Simpler windowing than Mikolov.
  void produceTrainingPairs(ProducerConsumerQueue<TrainWork> *pcq, bool *pDone);
  // Consume training pairs and update models.
  void consumeTrainingPairs(int threadId,
                            ProducerConsumerQueue<TrainWork> *pcq,
                            bool *pDone);
  // Same as sampleCorpusOnePair, but multithreaded.
  // Pays no attention to annotation spans.
  void sampleCorpus(const int numThreads = 20);

  // Focus "word" can be a word, phrase, or an annotated span.
  void produceTrainingPairsWithAnnotations(ProducerConsumerQueue<TrainWork> *pcq,
                                           bool *pDone);
  void sampleCorpusWithAnnotations(const int numThreads = 20);

  // Saves to Mikolov format, which is not good for reading back from GFS/CNS.
  void saveEmbeddingsMikolov();

  // Saves to numpy format with separate metadata file for matrix shape.
  virtual void saveEmbeddingsNumpy();
  // Loads from files in numpy format.
  virtual void loadEmbeddingsNumpy();

  // All in one saver.
  virtual void saveEmbeddings() {
    normalizeFocusEmbeddings();
    saveEmbeddingsMikolov();
    saveEmbeddingsNumpy();
  }
};   // class Word2vec


/**
 * Extension of Word2vec that embeds types to which entities belong.
 */
class Type2vec : public Word2vec {
 public:
  Type2vec(const Config &conf) : Word2vec(conf) { }
  /**
   * Loads KG entity-type memberships, upgrades vocabulary with types.
   * Requires all entities to be known to vocabulary.
   */
  void loadEntityToTypes(const string &dir);
  /**
   * Gets all containing types known to KG.
   * The first element of tids is set to eid itself.
   */
  void getContainingTypes(int32 eid, vector<int32> *tids);

  /**
   * Clears output before beginning.
   */
  void getDots(int32 contextTid, vector<int32> typeIds, vector<real> *dots);

  /**
   * Train only types as focus, but all types containing each entity annotated.
   * Wasteful because it reads and discards all other focus kinds.
   */
  void consumeTrainingPairsAllTypes(int thread,
                                    ProducerConsumerQueue<TrainWork> *pcq,
                                    bool *pDone);
  void sampleCorpusAllTypes(int numThreads = 20);

  /**
   * Training consumer changes to also take care of types.
   * This version lets each context word choose its own ent/type.
   * Uses hard max.
   */
  void consumeTrainingPairsHardMaxType(int threadId,
                                            ProducerConsumerQueue<TrainWork> *pcq,
                                            bool *pDone);

  void sampleCorpusHardMaxType();

 protected:
  hash_multimap<int32, int32> entToTypes;
};  // class Type2vec


}   // namespace ncc

#endif // _WORD2VEC_H_
