#include "img.h"
#include "util.h"

using namespace std;
using namespace cv;

void write_img(std::string filename, int c, int w, int h, float const *values) {
	assert(c == 3);
	vector<float> bla(values, values + c*w*h);
	//Mat mat(w, h, CV_32FC3, reinterpret_cast<void*>(const_cast<float*>(values)));

	float min(99999), max(-99999);
	for (size_t i(0); i < bla.size(); ++i) {
		if (bla[i] > max) max = bla[i];
		if (bla[i] < min) min = bla[i];
	}

	for (size_t i(0); i < bla.size(); ++i) {
	 	bla[i] = (bla[i] - min) / (max - min);
	//bla[i] *= 255.;
		//cout << bla[i] << " ";
	}
	///	cout << endl;

	Mat mat(h, w, CV_32FC3, reinterpret_cast<void*>(&bla[0]));
	mat *= 255.;
	mat.convertTo(mat, CV_8UC3);
	cvtColor(mat, mat, CV_RGB2BGR);

	if (!imwrite(filename, mat))
		throw StringException("Couldnt write img file");
}

void write_img1c(std::string filename, int w, int h, float const *values) {
	int c =1;
	vector<float> bla(values, values + c*w*h);
	//Mat mat(w, h, CV_32FC3, reinterpret_cast<void*>(const_cast<float*>(values)));

	// float min(99999), max(-99999);
	// for (size_t i(0); i < bla.size(); ++i) {
	// 	if (bla[i] > max) max = bla[i];
	// 	if (bla[i] < min) min = bla[i];
	// }

	float mean(0), std(0);
	for (size_t i(0); i < bla.size(); ++i) {
		mean += bla[i];
	}
	mean /= bla.size();

	for (size_t i(0); i < bla.size(); ++i) {
		std += (bla[i] - mean) * (bla[i] - mean);
	}

	std = sqrt(std / bla.size());
	float const eps(.0000001);
	float min = (mean - std*2);
	float max = (mean + std*2);

	for (size_t i(0); i < bla.size(); ++i) {
	 	float val = bla[i] = (bla[i] - min) / (max - min);
	 	val = val < 0. ? 0. : val;
	 	val = val > 1. ? 1. : val;

	//bla[i] *= 255.;
		//cout << bla[i] << " ";
	}
	///	cout << endl;

	Mat mat(h, w, CV_32F, reinterpret_cast<void*>(&bla[0]));
	mat *= 255.;
	mat.convertTo(mat, CV_8UC3);
	//cvtColor(mat, mat, CV_RGB2BGR);
	//cout << mat << endl;
	if (!imwrite(filename, mat))
		throw StringException("Couldnt write img file");
}


void write_img(std::string filename, int c, int w, int h, char *values) {
	assert(c == 3);
	vector<char> data(c*w*h);

	size_t n(0);
	for (size_t x(0); x < w; ++x)
		for (size_t y(0); y < h; ++y)
			for (size_t z(0); z < c; ++z, ++n) {
				//data[n] = int(values[z*w*h+y*w+x])+127;
				data[n] = values[z*w*h+y*w+x];
				//cout << int(data[n]) << " ";
			}
	//cout << endl;

	Mat mat(h, w, CV_8UC3, reinterpret_cast<void*>(reinterpret_cast<unsigned char*>(&data[0])));

	cout << mat.at<Vec3b>(Point(4, 4)) << endl;
	mat.convertTo(mat, CV_8UC3);
	cout << mat.at<Vec3b>(Point(4, 4)) << endl;
	cvtColor(mat, mat, CV_RGB2BGR);
	cout << mat.at<Vec3b>(Point(4, 4)) << endl;

	//cout << mat << endl;
	if (!imwrite(filename, mat))
		throw StringException("Couldnt write img file");
}

