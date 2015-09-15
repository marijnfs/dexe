#include "database.h"
#include "util.h"

#include "img.pb.h" //Img protobuf

#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace leveldb;
using namespace caffe;

template <typename T>
Database<T>::Database(string path) : N(0) {
	options.create_if_missing = true;
	Status status = DB::Open(options, path, &db);

	if (!status.ok()) {
		cerr << "Unable to open/create test database " << path << endl;
		cerr << status.ToString() << endl;
		return;
	}

	N = count();
}

template <typename T>
Database<T>::~Database() {
	delete db;
}

template <typename T>
void Database<T>::from_database(Database &other) {
	cout << "Copying database" << endl;
	Iterator* it = other.db->NewIterator(leveldb::ReadOptions());

	for (it->SeekToFirst(); it->Valid(); it->Next())
		db->Put(leveldb::WriteOptions(), it->key(), it->value());
	N = count();
}


/*
void Database::normalize_chw() {
	cout << "Normalizing dataset" << endl;
	Iterator* it = db->NewIterator(leveldb::ReadOptions());

	for (it->SeekToFirst(); it->Valid(); it->Next())
	{
		//cout << "|" << it->key().ToString() << "|" <<endl; //" : " << it->value().ToString() << endl;
		Datum datum;
		
		if (!datum.ParseFromString(it->value().ToString())) {
			cerr << "failed reading datum" << endl;
			return;
		}

		//Normalization
		vector<float> data(datum.data().size());
		unsigned char const *byte_data = reinterpret_cast<unsigned char const *>(datum.data().c_str());
		for (size_t i(0); i < data.size(); ++i)
			data[i] = float(byte_data[i]);

		normalize(&data);

		/*float mean(0);
		for (size_t i(0); i < data.size(); ++i) mean += data[i];
		mean /= data.size();
		for (size_t i(0); i < data.size(); ++i) data[i] -= mean;
		float std(0);
		for (size_t i(0); i < data.size(); ++i) std += data[i] * data[i];
		std = sqrt(std / (data.size() - 1));
		for (size_t i(0); i < data.size(); ++i) data[i] /= std;
		*/

		/*
		float min(99999), max(-99999);
		for (size_t i(0); i < data.size(); ++i) {
			if (data[i] > max) max = data[i];
			if (data[i] < min) min = data[i];
		}
		for (size_t i(0); i < data.size(); ++i)
			data[i] = 2.0 * (data[i] - min) / (max - min) - 1.0;
		*/

		//for (size_t i(0); i < data.size(); ++i) data[i] /= 256.;
		// CHW
/*
		int c = datum.channels();
		int h = datum.height();
		int w = datum.width();
		
		datum.clear_float_data();
		for (size_t i(0); i < data.size(); ++i) {
			//datum.add_float_data(data[(i * c) % (h * w * c) + (i / (w * h))]); // hwc to chw
 			//datum.add_float_data(data[(i * h * w) % (h * w * c) + (i / c)]); //chw to hwc
			datum.add_float_data(data[i]);
		}
		//cout << c << " " << h << " " << w << " " << data.size() << endl;

		string output;
		datum.SerializeToString(&output);
		db->Put(leveldb::WriteOptions(), it->key(), output);
	}
}
*/

template <typename T>
size_t Database<T>::count() {
	Iterator* it = db->NewIterator(leveldb::ReadOptions());

	size_t N(0);
	for (it->SeekToFirst(); it->Valid(); it->Next())
		++N;
	return N;
}

template <typename T>
string Database<T>::get_key(int index) {
	ostringstream oss;
	oss << setw(5) << setfill('0') << index;
	return oss.str();	
}

template <typename T>
T Database<T>::get_image(int index) {
	string data;
	leveldb::Slice key = get_key(index);
	if (!db->Get(ReadOptions(), key, &data).ok()) {
		cerr << "couldn't load key " << endl;
		exit(1);
	}

	T datum;
	if (!datum.ParseFromString(data)) {
		cerr << "couldn't parse data" << endl;
		exit(1);	
	}
	return datum;
}

template <typename T>
void Database<T>::add(T &datum) {
	string output;
	datum.SerializeToString(&output);
	db->Put(leveldb::WriteOptions(), get_key(N), output);
	++N;
}

template struct Database<caffe::Datum>;
template struct Database<Img>;
