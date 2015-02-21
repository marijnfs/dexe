#include "sampledatabase.h"
#include "util.h"

#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace leveldb;

SampleDatabase::SampleDatabase(string path) : N(0) {
	options.create_if_missing = true;
	Status status = DB::Open(options, path, &db);

	if (!status.ok()) {
		cerr << "Unable to open/create test database " << path << endl;
		cerr << status.ToString() << endl;
		return;
	}

	N = count();
	cout << "loaded " << N << " samples" << endl;
}

SampleDatabase::~SampleDatabase() {
	delete db;
}


size_t SampleDatabase::count() {
	Iterator* it = db->NewIterator(leveldb::ReadOptions());

	size_t N(0);
	for (it->SeekToFirst(); it->Valid(); it->Next())
		++N;
	return N;
}

string SampleDatabase::get_key(int index) {
	ostringstream oss;
	oss << index;
	return oss.str();	
}

Sample SampleDatabase::get_sample(int index) {
	string data;
	leveldb::Slice key = get_key(index);
	if (!db->Get(ReadOptions(), key, &data).ok()) {
		cerr << "couldn't load key " << endl;
		exit(1);
	}

	Sample sample;
	if (!sample.ParseFromString(data)) {
		cerr << "couldn't parse data" << endl;
		exit(1);	
	}
	return sample;
}

void SampleDatabase::add(Sample &sample) {
	string output;
	sample.SerializeToString(&output);
	db->Put(leveldb::WriteOptions(), get_key(N), output);
	++N;
}

