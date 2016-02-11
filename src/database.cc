#include "database.h"
// #include "util.h"

#include "img.pb.h" //Img protobuf
// #include "caffe.pb.h"

#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace leveldb;
// using namespace caffe;

Database::Database(string path) {
	options.create_if_missing = true;
	Status status = DB::Open(options, path, &db);

	if (!status.ok()) {
		cerr << "Unable to open/create test database " << path << endl;
		cerr << status.ToString() << endl;
		return;
	}

	index();
}

Database::~Database() {
	delete db;
}

void Database::clone_from_database(Database &other) {
	cout << "Copying database" << endl;
	Iterator* it = other.db->NewIterator(leveldb::ReadOptions());

	for (it->SeekToFirst(); it->Valid(); it->Next())
		db->Put(leveldb::WriteOptions(), it->key(), it->value());
	counts.clear();
	index();
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

void Database::index() {
	Iterator* it = db->NewIterator(leveldb::ReadOptions());
	cout << "reading database" << endl;
	size_t N(0);
	for (it->SeekToFirst(); it->Valid(); it->Next()) {
		string key = it->key().ToString();
		////hack
 	// 	std::string data;
  //   	db->Get(leveldb::ReadOptions(), key, &data);
		// db->Put(leveldb::WriteOptions(), key.substr(1), data);
		// db->Delete(leveldb::WriteOptions(), key);

		cout << "k: " << key << endl;
		int slash = key.find("/");
		if (slash < 0) continue;
		string name = key.substr(0, slash);
		if (counts.count(name) == 0)
			counts[name] = 0;
		counts[name]++;
	}
}

size_t Database::count(string name) {
	if (counts.count(name) == 0)
		counts[name] = 0;
	return counts[name];
}

string Database::get_key(string name, int index) {
	ostringstream oss;
	oss << name << "/" << setw(5) << setfill('0') << index;
	return oss.str();
}

/*
template <typename T>
T Database::load(string name, int index) {
	string data;
	leveldb::Slice key = get_key(name, index);
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
void Database::add(string name, T &datum) {
	int n = count(name);
	store(name, n, datum);
}

template <typename T>
void Database::store(string name, int index, T &datum) {
	string output;
	datum.SerializeToString(&output);
	db->Put(leveldb::WriteOptions(), get_key(name, index), output);
	counts[name]++;
}

//template struct Database<caffe::Datum>;
//template struct Database<Img>;

template <>
void Database::add<caffe::Datum>(string name, caffe::Datum &datum);

template <>
void Database::add<>(string name, Img &datum);

template <>
void Database::store<>(string name, int index, caffe::Datum &datum);

template <>
void Database::store<>(string name, int index, Img &datum);

template <>
caffe::Datum Database::load<caffe::Datum>(string name, int index);

template <>
Img Database::load<Img>(string name, int index);
*/
