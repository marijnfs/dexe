#ifndef __DATABASE_H__
#define __DATABASE_H__

#include <leveldb/db.h>
#include <string>
#include <map>
//read a caffe database


struct Database {
	leveldb::DB* db;
	leveldb::Options options;

	Database(std::string path);
	~Database();
	
	//caffe::Datum get_image(int index);
	//T get_image(int index);
	//void add(caffe::Datum &datum);

	template <typename T>
	void add(std::string, T &datum);

	template <typename T>
	void store(std::string, int index, T &datum);

	template <typename T>
	T load(std::string, int index);
	std::string get_key(std::string name, int index);


	void clone_from_database(Database &other);
	//void normalize_chw();
	size_t count(std::string);
	void index();

	std::map<std::string, int> counts;
};


//struct DatabaseRaw {
//	DatabaseRaw(std::string path);
//
//	
//};

#endif

