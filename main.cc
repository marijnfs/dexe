#include <cudnn.h>
#include <iostream>
#include <stdint.h>
#include "operations.h"
#include "util.h"
#include "database.h"
#include "network.h"

using namespace std;

void test() {
	TensorSet<float>x(1, 1, 4, 4);
	x.x.init_normal(1.0, .5);

	//ConvolutionOperation conv(1, 1, 4, 4);
	PoolingOperation<float> p(2, 2);
	TensorSet<float>out(p.output_shape(x.shape()));
	p.forward(x.x, out.x);
	cout << x.x.to_vector() << endl;
	cout << out.x.to_vector() << endl;

	out.grad.fill(-1);
	p.backward(x.x, out.x, out.grad, x.grad);
	cout << x.grad.to_vector() << endl;
	cout << out.grad.to_vector() << endl;
	

}

void test2() {
	//DataBase db("/home/marijnfs/dev/caffe/examples/cifar10/cifar10_train_leveldb");
	//DataBase db_test("/home/marijnfs/dev/caffe/examples/cifar10/cifar10_test_leveldb");
	//db_test.floatify();
	//Indices indices(db.N);

	double std(1.);
	double lr(.001);

	//Network network(TensorShape{1, 10, 1, 1});
	//network.add_conv(10, 1, 1);
	
	Network network(TensorShape{1, 3, 32, 32});
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
	DataBase db("/home/marijnfs/dev/caffe-rk/examples/cifar10/cifar10_train_leveldb");
	DataBase db_test("/home/marijnfs/dev/caffe-rk/examples/cifar10/cifar10_test_leveldb");
	Indices indices(db.N);

	db.normalize();
	db_test.normalize();

	double std(.05);
	double lr(.001);

	int n(1), c(3), h(32), w(32);
	int outc(10);
	Network network(TensorShape{n, c, w, h});
	network.add_conv(100, 2, 2);
	network.add_pool(2, 2);
	network.add_tanh();

	network.add_conv(200, 2, 2);
	network.add_pool(2, 2);
	network.add_tanh();
	
	network.add_conv(300, 2, 2);
	network.add_pool(2, 2);
	//network.add_conv(300, 1, 1);
	network.add_tanh();

	network.add_conv(400, 2, 2);
	network.add_pool(2, 2);
	//network.add_conv(400, 1, 1);
	network.add_tanh();

	network.add_conv(500, 2, 2);
	network.add_pool(2, 2);
	//network.add_conv(500, 1, 1);
	network.add_tanh();

	network.add_squash(100);
	network.add_tanh();

	network.add_squash(10);
	network.add_softmax();
	network.finish();
	//network.init_normal(0, std);
	network.init_uniform(std);

	//conv_operation1.init_normal(0, STD);
	//squash_operation.init_normal(0, STD);

	for (size_t e(0); e < 50000; ++e) {
		cout << "epoch: " << e << " lr: " << lr << endl;
		lr *= .99;
		Timer t;
		float err(0);
		int n_correct(0);

		indices.shuffle();
		for (size_t i(0); i < db.N; ++i) {
			//cout << i << endl;
			//caffe::Datum datum = db.get_image(49999);
			//for (size_t n(0); n < datum.float_data_size(); ++n)
			//		cout << datum.float_data(n) << " ";
			//cout << endl;

			//cout << "float data size: " << datum.float_data_size() << endl;
			
			caffe::Datum datum = db.get_image(indices[i]);
			const float *img_data = datum.float_data().data();

			//Tensor<float> t(1, 10, 1, 1);
			//t.init_normal(0.0, .005);
			//vector<float> x = t.to_vector();

			network.forward(img_data);
			//network.forward(&x[0]);
			network.calculate_loss(datum.label());
			network.backward();
			//vector<float> grad = network.gradient();

			//vector<float> fd_grad = network.fd_gradient(img_data, datum.label(), .00001);
			//vector<float> fd_grad = network.fd_gradient(&x[0], datum.label(), .001);
			//for (size_t n(0); n < grad.size(); ++n)
			//cout << (fd_grad[n] != 0 ? (grad[n] / fd_grad[n]) : grad[n]) << " " << grad[n] << ", ";
			//return 1;

			//network.l2(.000001);
			network.update(lr);

			err += network.loss();
			n_correct += network.n_correct();
			
			if (i % 5000 == 0) {
				vector<float> ov = network.output().to_vector();
				cout << i << " " << datum.label() << " " << ov << " " << ov[datum.label()] << endl;
			}
		}

		float test_err(0);
		int test_n_correct(0);
		
		for (size_t i(0); i < db_test.N; ++i) {
			caffe::Datum datum = db.get_image(indices[i]);
			const float *img_data = datum.float_data().data();

			network.forward(img_data);
			network.calculate_loss(datum.label());
				
			test_err += network.loss();
			test_n_correct += network.n_correct();
			
		}
		cout << "elapsed: " << t.since() << " err: " << (err / db.N) << " correct: " << n_correct << "/" << db.N << endl;
		cout << "test err: " << (test_err / db_test.N) << " correct: " << test_n_correct << "/" << db_test.N << endl;
	}
}

