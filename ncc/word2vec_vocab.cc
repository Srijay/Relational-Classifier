/**
 * Provides vocabulary and implements MikolovUnigramSampler methods.
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <boost/format.hpp>

// #include "base/integral_types.h"
// #include "base/logging.h"
// #include "file/base/file.h"
// #include "file/base/options.h"
// #include "file/base/recordio.h"

#include "ncc.pb.h"
#include "word2vec.h"
#include "seqzfile.h"

namespace ncc {

void MikolovUnigramSampler::load(const string &vocabRioPath) {

  assert( std::ifstream(vocabRioPath).good() );
  ncc::SeqzFileReader alphar1(vocabRioPath);
  //RecordReader alphar1(file::OpenOrDie(vocabRioPath, "r", file::Defaults()));
  std::string vwStr;
  VocabWord vw;
  int32 max_wid = 0, vocabSize = 0;
  int64 sumCount = 0;
  while (alphar1.get(&vwStr)) {
    const bool rc = vw.ParseFromString(vwStr); assert(rc);
    CHECK(vw.has_wid()) << "No wid in " << vw.DebugString();
    max_wid = std::max(max_wid, vw.wid());
    ++vocabSize;
    sumCount += vw.count();
  }
  alphar1.close();
  // Token ID space is expected to be dense.
  CHECK_EQ(1 + max_wid, vocabSize);
  vocab.resize(vocabSize);
  ncc::SeqzFileReader alphar2(vocabRioPath);
  bool slotMissing = false;
  int32 sumSlots = 0;
  while (alphar2.get(&vwStr)) {
    const bool rc = vw.ParseFromString(vwStr); assert(rc);
    CHECK(vw.has_wid()) << "No wid in " << vw.DebugString();
    CHECK(vw.has_word()) << "No word in " << vw.DebugString();
    CHECK(vw.has_count()) << "No count in " << vw.DebugString();
    if (vw.has_slots()) {
      sumSlots += vw.slots();
    }
    else {
      slotMissing = true;
    }
    vocab[vw.wid()].CopyFrom(vw);
    tokenToTidMap[TokenKind_Name(vw.kind()) + kindTokenSep + vw.word()] = vw.wid();
  }
  alphar2.close();
  cerr << "Loaded " << vocabSize << " tokens from " << vocabRioPath
      << "\n\tsumCount = " << sumCount
      << " sumSlots = " << sumSlots << endl;
  CHECK(!slotMissing) << "Missing slot/s";
  CHECK_GT(sumSlots, 0) << "All slots missing or zero";
  installSlots();
}

void MikolovUnigramSampler::installSlots() {
  // Check that token kinds are in enumeration order in vocabulary.
  int sumSlots = 0;
  TokenKind currentKind = TokenKind::Null;
  for (const auto &vw : vocab) {
    CHECK(vw.has_kind()) << vw.DebugString();
    CHECK(currentKind <= vw.kind()) << TokenKind_Name(currentKind)
        << " :: " << vw.DebugString();
    CHECK(vw.has_slots()) << vw.DebugString();
    currentKind = vw.kind();
    sumSlots += vw.slots();
  }
  wordPicker.resize(sumSlots);
  currentKind = TokenKind::Null;
  int base = 0;
  for (int wid = 0; wid < vocab.size(); ++wid) {
    const auto &vw = vocab[wid];
    CHECK(vw.has_slots() && vw.has_kind()) << vw.DebugString();
    for (int sx = 0; sx < vw.slots(); ++sx) {
      wordPicker[base] = wid;
      if (currentKind < vw.kind()) {
        kindToSlotEnd[currentKind] = base;
        kindToSlotBegin[vw.kind()] = base;
      }
      ++base;
      currentKind = vw.kind();
    }
  }
  kindToSlotEnd[currentKind] = base;
  iunif = std::uniform_int_distribution<int>(0, wordPicker.size()-1);
  cerr << "Installed " << sumSlots << " slots\n";
  for (const auto &kk : kindToSlotBegin) {
    cerr << "\t" << TokenKind_Name(kk.first) << " --> " << kk.second
        << " ... " << kindToSlotEnd[kk.first] << endl;
  }
  saveSlots("/tmp/test_slots.txt");
}

void MikolovUnigramSampler::redoSlots(int32 newNumSlots, double unipower) {
  uint64 sumCount = 0;
  double trainWordsPow = 0;
  for (auto &vw : vocab) {
    CHECK(vw.has_count()) << vw.DebugString();
    sumCount += vw.count();
    trainWordsPow += std::pow(vw.count(), unipower);
    vw.set_slots(0);
  }
  wordPicker.clear();
  cerr << "RedoSlots sumCount=" << sumCount <<
      " trainWordsPow=" << trainWordsPow << endl;
  int wid = 0;
  double d1 = std::pow(vocab[wid].count(), unipower) / trainWordsPow;
  for (int a = 0; a < newNumSlots; a++) {
    auto &vw = vocab[wid];
    vw.set_slots(1 + vw.slots());
    if (( 1.0 * a / newNumSlots ) > d1) {
      wid++;
      d1 += std::pow(vocab[wid].count(), unipower) / trainWordsPow;
    }
    if (wid >= vocab.size()) wid = vocab.size() - 1;
  }
  installSlots();
}

void MikolovUnigramSampler::save(const string &ariop) {
  ncc::SeqzFileWriter alphaw(ariop);
  for (const auto &vw : vocab) {
    alphaw.put(vw.SerializeAsString());
  }
  alphaw.close();
}

void MikolovUnigramSampler::saveReadable(const string &alphaTxtPath) {
  std::ofstream alphaTxtOfs;  // also dump alphabet in text format
  alphaTxtOfs.open(alphaTxtPath.c_str());
  for (const auto &vw : vocab) {
    // Cord alphaLine;
    // alphaLine.AppendF("%s %g %d %d\n", vw.word().c_str(),
    //                   vw.count(), vw.slots(), vw.wid());
    alphaTxtOfs << format("%s %g %d %d", vw.word().c_str(), vw.count(),
			  vw.slots(), vw.wid());
  }
  alphaTxtOfs.close();
}

void MikolovUnigramSampler::saveSlots(const string &slotPath) {
  std::ofstream slotOfs(slotPath.c_str());
  int32 prevTid = -1;
  for (int slot = 0; slot < wordPicker.size(); ++slot) {
    if (wordPicker[slot] != prevTid) {
      slotOfs << slot << ' ' << wordPicker[slot] << endl;
    }
    prevTid = wordPicker[slot];
  }
  slotOfs.close();
  cerr << "Saved unigram slots to " << slotPath << endl;
}

bool MikolovUnigramSampler::tidToToken(int tid,
                                       TokenKind *kind, string *token) const
{
  CHECK_LE(0, tid);
  CHECK_LT(tid, vocab.size());
  const auto &vw = vocab[tid];
  CHECK(vw.has_kind() && vw.has_word()) << vw.DebugString();
  *kind = vw.kind();
  token->assign(vw.word());
  return true;
}

int MikolovUnigramSampler::tokenToTid(TokenKind kind,
                                      const string &token) const
{
  const string key = TokenKind_Name(kind) + kindTokenSep + token;
  const auto kx = tokenToTidMap.find(key);
  if (kx == tokenToTidMap.end()) return -1;
  CHECK_LE(0, kx->second);
  CHECK_LT(kx->second, vocab.size());
  return kx->second;
}

int32 MikolovUnigramSampler::sample(uint64 rn, TokenKind kind) {
  const auto slot =
      (kind == TokenKind::Null)?
          (rn % wordPicker.size()) :
          (kindToSlotBegin[kind] +
              rn % (kindToSlotEnd[kind] - kindToSlotBegin[kind]) );
  CHECK_LE(0, slot);
  CHECK_LT(slot, wordPicker.size());
  return wordPicker[slot];
}

}   // namespace ncc
