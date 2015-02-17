#ifndef __SAMPLE_DATABASE_H__
#define __SAMPLE_DATABASE_H__

#include <leveldb/db.h>
#include "sample.pb.h"
#include <string>

//read a caffe database
struct SampleDatabase {
	leveldb::DB* db;
	leveldb::Options options;

	SampleDatabase(std::string path);
	~SampleDatabase();
	
	Sample get_sample(int index);
	std::string get_key(int index);
	void add(Sample &sample);
	size_t count();

	size_t N;
};


//struct SampleDatabaseRaw {
//	SampleDatabaseRaw(std::string path);
//
//	
//};

#endif

