#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

// #include "base/init_google.h"
// #include "base/integral_types.h"
// #include "base/logging.h"
// #include "file/base/file.h"
// #include "file/base/filelineiter.h"
// #include "file/base/options.h"
// #include "file/base/recordio.h"

#include "ncc.pb.h"
#include "word2vec.h"
#include "seqzfile.h"

extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size);

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char dumpcv_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING], save_unigram_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1, use_position = 0;
int *vocab_hash;
int vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0, dumpcv = 0;
int iter = 5;
real alpha = 0.025, starting_alpha, sample = 0;
const real unipower = 0.75;
real *syn0, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
  int a, i;
  float sumRawCount=0, sumFlatCount=0;
  for (a = 0; a < vocab_size; a++) {
    sumRawCount += vocab[a].cn;
    sumFlatCount += pow(vocab[a].cn, unipower);
  }
  fprintf(stderr, "vocabSize=%d sumRawCount=%lld sumFlatCount=%lld\n",
	  vocab_size, (long long) sumRawCount, (long long) sumFlatCount);
  table = (int *)malloc(table_size * sizeof(int));
  i = 0;
  real d1 = pow(vocab[i].cn, unipower) / sumFlatCount;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      auto prev = pow(vocab[i].cn, unipower) / sumFlatCount;
      if (debug_mode > 11) {
        printf("UNI %d %lld %lld %lld %g\n",
               i, vocab[i].cn, (long long) sumRawCount,
               (long long) sumFlatCount, prev);
      }
      i++;
      d1 += pow(vocab[i].cn, unipower) / sumFlatCount;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

void SaveUnigramTable(const char *uname) {
  std::map<int, int> wid_to_slots;
  for (int slot = 0; slot < table_size; ++slot) {
    ++wid_to_slots[ table[slot] ];
  }
  ncc::SeqzFileWriter alphaw(uname);
  ncc::VocabWord vw;
  for (int wid = 0; wid < vocab_size; ++wid) {
    vw.Clear();
    vw.set_wid(wid);
    vw.set_word(vocab[wid].word);
    vw.set_count(vocab[wid].cn);
    vw.set_kind(ncc::TokenKind::Token);
    vw.set_slots(wid_to_slots[wid]);
    alphaw.put(vw.SerializeAsString());
  }
  alphaw.close();
}

void FillSigmoid() {
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (int i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
}

real EvalSigmoid(real dot) {
  if (dot > MAX_EXP) return 1;
  if (dot < -MAX_EXP) return 0;
  return expTable[(int)((dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  const int diff0 = ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
  if (diff0 != 0) return diff0;
  return strcmp(((struct vocab_word *)b)->word, ((struct vocab_word *)a)->word);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  fprintf(stdout, "ALERT ReduceVocab\n");
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %d\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %d\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  posix_memalign((void **)&syn0, (size_t) 128, (size_t) (vocab_size * layer1_size * sizeof(real)));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  // hs == false
  if (negative>0) {
    {  // use_position == false
      posix_memalign((void **)&syn1neg, (size_t) 128, (size_t) (vocab_size * layer1_size * sizeof(real)));
      if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
      for (int b = 0; b < layer1_size; b++) {
	for (int a = 0; a < vocab_size; a++) {
	  syn1neg[a * layer1_size + b] = 0;
	}
      }
    }
  }
  for (int b = 0; b < layer1_size; b++) {
    for (int a = 0; a < vocab_size; a++) {
      syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
    }
  }
  CreateBinaryTree();
}

#define W2V_TAPER 0
#define W2V_SAMPLE 0
#define W2V_RANDOM 0

void *TrainModelThread(void *id) {
  ncc::SamplerHash samplerHash;
  int64_t sentence_count = 0, sentence_length = 0, sentence_position = 0;
  int64_t word_count = 0, last_word_count = 0;
  int32_t sen[MAX_SENTENCE_LENGTH + 1];
  int local_iter = iter;
  const time_t ttx = time(0);
  uint64_t next_random = (uint64_t)id + ttx;
  fprintf(stderr, "Thread=%p next_random = %" PRIu64 "\n", id, next_random);
  clock_t now;
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        fprintf(stderr, "%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
               word_count_actual / (real)(iter * train_words + 1) * 100,
               word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
#if W2V_TAPER
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
#else
      alpha = starting_alpha;
#endif
    }
    if (sentence_length == 0) {
      while (1) {
        const long long word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          CHECK(W2V_SAMPLE) << "Inconsistent sample flag.";
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          ncc::SamplerHash::bump(&next_random);
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
      ++sentence_count;
    }
//    if (feof(fi)) break;
//    if (word_count > train_words / num_threads) break;
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      cerr << id << " reduced local_iter to " << local_iter << endl;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    const auto word = sen[sentence_position];
    if (word == -1) continue;
#if W2V_RANDOM
    ncc::SamplerHash::bump(&next_random);
    const int shrunk = next_random % window;
#else
    const int shrunk = samplerHash.hashToWindowSize(sentence_count, sentence_length, sentence_position) % window;
#endif
    CHECK_LE(0, shrunk);
    CHECK_LT(shrunk, window);
    //train skip-gram
    for (int a = shrunk; a < window * 2 + 1 - shrunk; a++) {
      if (a == window) continue;
      const long long c = sentence_position - window + a;
      if (c < 0) continue;
      if (c >= sentence_length) continue;
      const auto last_word = sen[c];
      assert(last_word != -1);
      assert(last_word != 0);
      if (last_word <= 0) continue;
      const auto last_word_row = last_word * layer1_size;
      for (int x01c = 0; x01c < layer1_size; x01c++) neu1e[x01c] = 0; 	  // neu1e[:] = 0
      // NEGATIVE SAMPLING
      if (negative > 0) {
        int target;
        real label;
        for (int x02d = 0; x02d < negative + 1; x02d++) {
          if (x02d == 0) {
            target = word;
            label = 1;    // positive example
            cout << "SEN " << sentence_count << endl;
            ncc::Word2vec::reportPair(sentence_count, sentence_length, sentence_position, c, last_word, target, x02d, 0, label);
          }
          else {
#if W2V_RANDOM
            const auto rv = ncc::SamplerHash::bump(&next_random);
            target = table[(rv >> 16) % table_size];
#else
            const auto rv = samplerHash.hashToNegativeWord(sentence_count, sentence_length, sentence_position, c, x02d);
            target = table[rv % table_size];
#endif
            if (target == 0) target = 1 + rv % (vocab_size - 1);
            if (target == word) continue;
            label = 0;    // negative example
            ncc::Word2vec::reportPair(sentence_count, sentence_length, sentence_position, c, last_word, target, x02d, rv % table_size, label);
          }
          const long long target_word_row = target * layer1_size;
          real dot = 0;
          for (int x05c = 0; x05c < layer1_size; x05c++) {
            dot += syn0[x05c + last_word_row] * syn1neg[x05c + target_word_row];
          }
          // dot = syn0[last_word, :] * syn1neg[target_word, :]
          // predicted_label = sigmoid(dot)
          const real g = (label - EvalSigmoid(dot)) * alpha;
          for (int x06c = 0; x06c < layer1_size; x06c++) neu1e[x06c] += g * syn1neg[x06c + target_word_row];
          for (int x07c = 0; x07c < layer1_size; x07c++) syn1neg[x07c + target_word_row] += g * syn0[x07c + last_word_row];
        } // for one positive and some negative samples
      } // if negative sampling
      // Learn weights input -> hidden
      for (int x08c = 0; x08c < layer1_size; x08c++) syn0[x08c + last_word_row] += neu1e[x08c];
    } // for a = sweep

    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  } // loop forever
  fclose(fi);
  free(neu1e);
  pthread_exit(NULL);
}

void SaveSlots(const string &slotPath) {
  cerr << "Saving slots to " << slotPath << endl;
  ofstream slotOfs(slotPath.c_str());
  int32 prevTid = -1;
  for (int slot = 0; slot < table_size; ++slot) {
    if (table[slot] != prevTid) {
      slotOfs << slot << ' ' << table[slot] << endl;
    }
    prevTid = table[slot];
  }
  slotOfs.close();
}


void TrainModel() {
  FILE *fo;
  FILE *fo2;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) {
    InitUnigramTable();
    if (save_unigram_file[0] != 0) SaveUnigramTable(save_unigram_file);
    SaveSlots("/tmp/gold_slots.txt");
  }
  fprintf(stderr, "Starting threads\n");
  start = clock();
  for (long long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (long long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  if (dumpcv_file[0] != 0) fo2 = fopen(dumpcv_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%d %d\n", vocab_size, layer1_size);
    if (dumpcv_file[0] != 0) fprintf(fo2, "%d %d\n", vocab_size, layer1_size);
    for (int a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (int b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (int b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
      if (dumpcv_file[0] != 0) {
         fprintf(fo2, "%s ", vocab[a].word);
         if (binary) for (int b = 0; b < layer1_size; b++) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo2);
         else for (int b = 0; b < layer1_size; b++) fprintf(fo2, "%lf ", syn1neg[a * layer1_size + b]);
         fprintf(fo2, "\n");
      }
    }
  } else {
    // Run K-means on the word vectors
    // Save the K-means classes
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int argc_dummy = 1;
  //char *argv_fake[] = { argv[0] };
  //char **argv_dummy = &argv_fake[0];
  //InitGoogle(argv[0], &argc_dummy, &argv_dummy, false);

  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-save-unigram <file>\n");
    printf("\t\tThe unigram table will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");
    printf("\t-dumpcv <filename>\n");
    printf("\t\tDump the context vectors, in file <filename>\n");
    printf("\t-pos 1\n");
    printf("\t\tInclude sequence position information in context.\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }
  output_file[0] = 0;
  dumpcv_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-unigram", argc, argv)) > 0) strcpy(save_unigram_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-dumpcv", argc, argv)) > 0) strcpy(dumpcv_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-pos", argc, argv)) > 0) use_position = 1;
  if (dumpcv_file[0] != 0 && negative == 0) {
     printf("-dumpcv requires negative training.\n\n");
     return 0;
  };
  if (dumpcv_file[0] != 0 && (use_position > 0)) {
     printf("-dumpcv cannot run with use_position yet.\n\n");
     return 0;
  };
  if ((hs > 0 || cbow > 0) && (use_position > 0)) {
     printf("-use_position require skip-gram negative-sampling training.\n\n");
     return 0;
  };
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  FillSigmoid();
  TrainModel();
  return 0;
}
