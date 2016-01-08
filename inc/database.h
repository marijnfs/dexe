#ifndef __DATABASE_H__
#define __DATABASE_H__

#include <leveldb/db.h>
#include <string>
#include <iostream>
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
  T load(std::string name, int index) {
    std::string data;
    leveldb::Slice key = get_key(name, index);
    if (!db->Get(leveldb::ReadOptions(), key, &data).ok()) {
      std::cerr << "couldn't load key " << std::endl;
      exit(1);
    }

    T datum;
    if (!datum.ParseFromString(data)) {
      std::cerr << "couldn't parse data" << std::endl;
      exit(1);	
    }
    return datum;
  }

  template <typename T>
  void add(std::string name, T &datum) {
    int n = count(name);
    store(name, n, datum);
  }
  
  template <typename T>
  void store(std::string name, int index, T &datum) {
    std::string output;
    datum.SerializeToString(&output);
    db->Put(leveldb::WriteOptions(), get_key(name, index), output);
    counts[name]++;
  }
  
  /*	template <typename T>
	void add(std::string, T &datum);

	template <typename T>
	void store(std::string, int index, T &datum);

	template <typename T>
	T load(std::string, int index);
  */

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

