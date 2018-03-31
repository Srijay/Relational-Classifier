#ifndef __SEQZFILE__
#define __SEQZFILE__

#include <fstream>
#include <cstdint>
#include "gzstream.h"

namespace ncc {

  using std::cerr;
  using std::endl;

  class SeqzFileWriter {
  protected:
    ogzstream gos;

  public:
  SeqzFileWriter(const std::string &path): gos(path.c_str()) { }

    void put(const std::string &rec) {
      int32_t len = rec.size();
      gos.write((char*) &len, sizeof(len));
      gos.write(rec.c_str(), len);
      gos.write((char*) &len, sizeof(len));
    }

    void close() {
      gos.close();
    }
  };
  
  class SeqzFileReader {
  protected:
    igzstream gis;
    
  public:
  SeqzFileReader(const std::string &path) : gis(path.c_str()) { }
    
    bool get(std::string *rec) {
      int32_t bytesToRead = 0;
      gis.read((char*) &bytesToRead, sizeof(bytesToRead) );
      rec->clear();
      while (bytesToRead-- > 0) {
	char ch;
	gis.read(&ch, sizeof(ch));
	rec->append(1, ch);
      }
      int32_t bytesWeRead = 0;
      gis.read((char*) &bytesWeRead, sizeof(bytesWeRead));
      return rec->size() == bytesWeRead and bytesWeRead > 0;
    }

    void close() {
      gis.close();
    }
  };
}  // namespace ncc


/*

Test harness:

#include <cstdio>
#include "gzstream.h"
#include "seqzfile.h"

int main() {
  std::string tf = std::tmpnam(nullptr);
  
  ncc::SeqzFileWriter sfw(tf);
  for (auto word : { "row", "your", "boat" }) {
    sfw.put(word);
  }
  sfw.close();
  
  ncc::SeqzFileReader sfr(tf);
  std::string rec;
  int64_t nrec = 0;
  while (sfr.get(&rec)) {
    std::cout << rec << std::endl;
    ++nrec;
  }
  sfr.close();
  std::cerr << "Read " << nrec << " records from " << tf << std::endl;
  return 0;
}

 */


#endif // __SEQZFILE__
