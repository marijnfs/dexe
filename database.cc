#include "database.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace leveldb;
using namespace caffe;

DataBase::DataBase(string path) {
	options.create_if_missing = true;
	Status status = DB::Open(options, path, &db);

	if (!status.ok()) {
		cerr << "Unable to open/create test database './testdb'" << endl;
		cerr << status.ToString() << endl;
		return;
	}

}

void DataBase::loop() {
	Iterator* it = db->NewIterator(leveldb::ReadOptions());

	for (it->SeekToFirst(); it->Valid(); it->Next())
	{
		cout << it->key().ToString() << endl; //" : " << it->value().ToString() << endl;
		Datum datum;
		
		if (!datum.ParseFromString(it->value().ToString())) {
			cerr << "failed reading datum" << endl;
			return;
		}
	}
}

caffe::Datum DataBase::get_image(int index) {
	ostringstream oss;
	oss << setw(5) << setfill('0') << index << endl;

	string data;
	if (db->Get(ReadOptions(), oss.str(), &data).ok()) {
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