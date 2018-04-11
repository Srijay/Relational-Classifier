all: ncc_pb2.py ncc.pb.o gzstream.o \
word2vec_base.o word2vec_vocab.o word2vec_gold.o word2vec_main.o word2vec_main.exe

CPPFLAGS=-std=c++11 -O3 -pthread # -Wno-deprecated
LIBS=-lz -lssl -lcrypto -lboost_system -lboost_iostreams -lprotobuf -pthread -Wl,--no-as-needed
HEADERS=gzstream.h seqzfile.h pcqueue.h threadpool.h \
ncc.pb.h word2vec.h word2vec_loss.h

ncc_pb2.py: ncc.proto
	protoc --python_out . ncc.proto

ncc.pb.h: ncc.proto
	protoc --cpp_out . ncc.proto
ncc.pb.cc: ncc.proto
	protoc --cpp_out . ncc.proto

word2vec_main.exe: word2vec_main.o gzstream.o ncc.pb.o word2vec_base.o word2vec_vocab.o $(HEADERS)
	g++ $(CPPFLAGS) -o $@ $^ $(LIBS)

%.o : %.cc $(HEADERS)
	g++ -c $(CFLAGS) $(CPPFLAGS) $< -o $@

ncc.pb.o : ncc.pb.cc $(HEADERS)
	g++ -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	-rm *.exe *.o *~ ncc_pb2.py ncc.pb.h ncc.pb.cc


# https://github.com/ben-strasser/fast-cpp-csv-parser
# https://github.com/PeterScott/murmur3
