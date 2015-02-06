#ifndef __DATABASE_H__
#define __DATABASE_H__

#include "leveldb/db.h"
#include "caffe.pb.h"
#include <string>

//read a caffe database
struct Database {
	leveldb::DB* db;
	leveldb::Options options;

	Database(std::string path);
	~Database();
	
	caffe::Datum get_image(int index);
	std::string get_key(int index);
	void add(caffe::Datum &datum);


	void from_database(Database &other);
	void normalize_chw();
	size_t count();

	size_t N;
};


//struct DatabaseRaw {
//	DatabaseRaw(std::string path);
//
//	
//};

#endif

