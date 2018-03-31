#include <iostream>
#include <string>
#include <getopt.h>

#include "ncc.pb.h"
#include "word2vec.h"

int main(int argc, char** argv) {

  string in, out;
  int window=5, neg=5, epochs=5, threads=20;
  std::size_t dim = 100;
  float unipower=0.75, step=0.025;
  bool redo=false, warm=false;
  enum {
    IN = 1001, UNIPOWER, DIM, WINDOW, NEG, STEP,
    EPOCHS, REDO, WARM, THREADS, OUT,
  };
  struct option long_options[] = {
    {"in", required_argument, 0, IN},
    {"unipower", optional_argument, 0, UNIPOWER},
    {"dim", optional_argument, 0, DIM},
    {"window", optional_argument, 0, WINDOW},
    {"neg", optional_argument, 0, NEG},
    {"step", optional_argument, 0, STEP},
    {"epochs", optional_argument, 0, EPOCHS},
    {"redo", optional_argument, 0, REDO},
    {"warm", optional_argument, 0, WARM},
    {"threads", optional_argument, 0, THREADS},
    {"out", required_argument, 0, OUT},
    {0, 0, 0, 0},
  };
  auto conf = ncc::Config();
  int long_index = -1;
  while (getopt_long_only(argc, argv, "", long_options, &long_index) != -1) {
    
    std::cerr << long_options[long_index].val << ' '
	      << ( optarg? optarg : "nullptr" ) << std::endl;
    switch (long_options[long_index].val) {
    case IN:
      in = optarg;
      break;
    case UNIPOWER:
      unipower = atof(optarg);
      conf.set_unipower(unipower);
      break;
    case DIM:
      dim = atoi(optarg);
      conf.set_dim(dim);
      break;
    case WINDOW:
      window = atoi(optarg);
      conf.set_window(window);
      break;
    case NEG:
      neg = atoi(optarg);
      conf.set_neg(neg);
      break;
    case STEP:
      step = atof(optarg);
      conf.set_step(step);
      break;
    case EPOCHS:
      epochs = atof(optarg);
      conf.set_epochs(epochs);
      break;
      case OUT:
      out = optarg;
      break;
    default:
      std::cerr << "Warning long_index=" << long_index << std::endl;
    }
  }

  // TODO(soumenc) override defaults from command line.
  cerr << "main conf = [[ " << conf.DebugString() << " ]]\n";

  if (conf.epochs() > 0) {
    ncc::Word2vec w2v(conf);

    w2v
      .withDirs(in, out)
      .construct() 
      .checkInputSanity() 
    ;

#define ANNOT 1 

    for (int epoch = 0; epoch < conf.epochs(); ++epoch) {
      cerr << "### EPOCH " << epoch << " word/phrase/ent ####\n";
      if (warm or epoch > 0) w2v.loadEmbeddingsNumpy();
#if ANNOT
      w2v.sampleCorpusWithAnnotations(threads); 
#else
      //  w2v.sampleCorpusMikolov();
      w2v.sampleCorpusOnePair();
#endif
    }
  }

  if (false) {
    cerr << "Training types\n";
    ncc::Type2vec t2v(conf);
    t2v.withDirs(in, in).construct();
    t2v.loadEntityToTypes(in);
    if (warm) t2v.loadEmbeddingsNumpy();
    t2v.sampleCorpusAllTypes(threads);
  }
}
