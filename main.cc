#include <cudnn.h>
#include <iostream>
#include <stdint.h>
#include "layers.h"
#include "util.h"
#include "database.h"

using namespace std;

int main() {
	DataBase db("/home/marijnfs/dev/caffe/examples/cifar10/cifar10_train_leveldb");
	// db.floatify();
	caffe::Datum datum = db.get_image(2);

	cout << "n data: " << datum.data().size() << endl;
	cout << "n float data: " << datum.float_data_size() << endl;
	cout << "width: " << datum.width() << endl;
	double STD(.5);

	int n(1), c1(3), c2(50), h(32), w(32);

	Tensor h1(n, c1, h, w);

	Tensor h2(n, c2, h, w);
	Tensor h3(n, c2, h, w);

	int outc(10);
	Tensor o1(n, outc, 1, 1), o(n, outc, 1, 1);


	ConvolutionLayer conv_layer1(c1, c2, 4, 4);
	conv_layer1.init_normal(0, STD);

	TanhLayer tanh_layer;
	SquashLayer squash_layer(h3, 10);
	SoftmaxLayer softmax;

	Timer t;

	const float *img_data = datum.float_data().data();

	h1.from_ptr(img_data);
	
	conv_layer1.forward(h1, h2);
	tanh_layer.forward(h2, h3);
	squash_layer.forward(h3, o1);
	softmax.forward(o1, o);

	Tensor answer(n, outc, 1, 1);
	
	cout << h2.to_vector() << endl;
	// vector<float> vec2 = t2_map.to_vector();
	// vec2.resize(10);
	// cout << "t2out" << vec2 << endl << vec2 << endl;

	// Tensor out(n, w2, h2, 2);
	cout << "elapsed: " << t.since() << endl;
}
