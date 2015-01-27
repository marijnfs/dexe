#include "leveldb/db.h"
#include "caffe.pb.h"
#include <string>

//read a caffe database
struct DataBase {
	leveldb::DB* db;
	leveldb::Options options;

	DataBase(std::string path);

	caffe::Datum get_image(int index);
	void loop();
};


