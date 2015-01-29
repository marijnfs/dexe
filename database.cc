#include "database.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace leveldb;
using namespace caffe;

DataBase::DataBase(string path) : N(0) {
	options.create_if_missing = true;
	Status status = DB::Open(options, path, &db);

	if (!status.ok()) {
		cerr << "Unable to open/create test database './testdb'" << endl;
		cerr << status.ToString() << endl;
		return;
	}

	N = count();
}

DataBase::~DataBase() {
	delete db;
}

void DataBase::floatify() {
	Iterator* it = db->NewIterator(leveldb::ReadOptions());

	for (it->SeekToFirst(); it->Valid(); it->Next())
	{
		cout << "|" << it->key().ToString() << "|" <<endl; //" : " << it->value().ToString() << endl;
		Datum datum;
		
		if (!datum.ParseFromString(it->value().ToString())) {
			cerr << "failed reading datum" << endl;
			return;
		}


		datum.clear_float_data();
		vector<float> data(datum.data().size());
		for (size_t i(0); i < data.size(); ++i) {
			// cout << int(datum.data()[i]) << endl;
			float val = float(datum.data()[i]) / (255 / 2);
			datum.add_float_data(val);
			// cout << data[i] << " ";
		}
		string output;
		datum.SerializeToString(&output);
		db->Put(leveldb::WriteOptions(), it->key(), output);
	}
}

size_t DataBase::count() {
	Iterator* it = db->NewIterator(leveldb::ReadOptions());

	size_t N(0);
	for (it->SeekToFirst(); it->Valid(); it->Next())
		++N;
	return N;
}

caffe::Datum DataBase::get_image(int index) {
	ostringstream oss;
	oss << setw(5) << setfill('0') << index;

	string data;
	leveldb::Slice key = oss.str();
	if (!db->Get(ReadOptions(), key, &data).ok()) {
		cerr << "couldn't load key " << oss.str() << endl;
		exit(1);
	}

	Datum datum;
	if (!datum.ParseFromString(data)) {
		cerr << "couldn't parse data" << endl;
		exit(1);	
	}
	return datum;
}
