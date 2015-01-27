#include <cudnn.h>
#include <iostream>
#include <stdint.h>
#include "layers.h"
#include "util.h"
#include "database.h"

using namespace std;

int main() {
	DataBase db("/home/marijnfs/dev/caffe/examples/cifar10/cifar10_train_leveldb");
	caffe::Datum datum = db.get_image(2);

	double STD(.005);

	int n(1), c1(2), c2(50), h(480), w(640);

	Tensor h1(n, c1, h, w);
	Tensor h2(n, c2, h, w);
	Tensor h3(n, c2, h, w);
	Tensor o(n, c2, h, w);


	ConvolutionLayer conv_layer1(c1, c2, 4, 4);
	conv_layer1.init_normal(0, STD);

	TanhLayer tanh_layer;
	SoftmaxLayer softmax;

	Timer t;

	conv_layer1.forward(h1, h2);
	tanh_layer.forward(h2, h3);

	// vector<float> vec2 = t2_map.to_vector();
	// vec2.resize(10);
	// cout << "t2out" << vec2 << endl << vec2 << endl;

	// Tensor out(n, w2, h2, 2);
	cout << "elapsed: " << t.since() << endl;
}

