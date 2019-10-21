#include <cudnn.h>
#include <iostream>
#include <stdint.h>
#include "operations.h"
#include "util.h"
#include "database.h"
#include "network.h"
#include "img.h"
#include "colour.h"

#include <unistd.h>
#include <ctime>
#include <cuda.h>

using namespace std;

void test3() {
	cout << "Test 3" << endl;

	Network<float> net;
	Node node;
	auto node2 = net.convolution(3)(node);
	cout << "Done" << endl;
}

void test1() {
	Network<float> network(TensorShape{1, 3, 640, 480});

	network.add_conv(32, 3, 3);
	network.add_pool(2, 2);
	network.add_relu();

	network.add_conv(64, 3, 3);
	network.add_pool(2, 2);
	network.add_relu();


	network.add_conv(128, 3, 3);
	network.add_pool(2, 2);
	network.add_relu();

	network.add_conv(256, 3, 3);
	network.add_relu();
	network.add_pool(2, 2);

	network.add_conv(640, 3, 3);
	network.add_relu();
	network.add_pool(2, 2);

	network.add_softmax();
	network.finish();

	Timer timer;
	network.input().init_normal(1.0, .5);
	network.forward();
	cudaDeviceSynchronize();
	cout << timer.since() << endl;
}

void test2() {
	//Database db("/home/marijnfs/dev/caffe/examples/cifar10/cifar10_train_leveldb");
	//Database db_test("/home/marijnfs/dev/caffe/examples/cifar10/cifar10_test_leveldb");
	//db_test.floatify();
	//Indices indices(db.N);

	double std(1.);
	double lr(.002);

	//Network network(TensorShape{1, 10, 1, 1});
	//network.add_conv(10, 1, 1);

	Network<float> network(TensorShape{1, 3, 32, 32});
	Tensor<float> data(TensorShape{1, 3, 32, 32});

	data.init_normal(1.0, 1.0);
	vector<float> bla = data.to_vector();

	network.add_conv(10, 3, 3);
	network.add_pool(4, 4);
	//network.add_conv(100, 1, 1);
	network.add_tanh();

	//network.add_conv(10, 2, 2);
	//network.add_pool(2, 2);
	//network.add_conv(200, 1, 1);
	//network.add_tanh();

	network.add_squash(10);
	//network.add_softmax();
	network.finish();
	network.init_normal(0, std);

	//caffe::Datum datum = db.get_image(0);
	//const float *img_data = datum.float_data().data();
	const float *img_data = &bla[0];

	int label = 5;
	network.forward(img_data);
	network.calculate_loss(label);
	network.backward();

	vector<float> grad = network.gradient();
	cout << grad.size() << endl;
	cout << network.output().to_vector() << endl;

	vector<float> fd_grad = network.fd_gradient(img_data, label, .01);
	cout << fd_grad.size() << endl;

	for (size_t i(0); i < grad.size(); ++i)
		if (abs<float>(grad[i] + fd_grad[i]) > .001)
			cout << i << "\t" << grad[i] << "\t" << fd_grad[i] << "\t" << abs<float>(grad[i] + fd_grad[i]) << "\t" << (abs<float>(grad[i] + fd_grad[i]) / abs<float>(grad[i]))  << endl;
}


int main() {
	test3();
}

