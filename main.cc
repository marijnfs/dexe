#include <cudnn.h>
#include <iostream>
#include <stdint.h>
#include "operations.h"
#include "util.h"
#include "database.h"
#include "network.h"

using namespace std;

void test() {
	TensorSet x(1, 1, 1, 1);
	x.x.fill(2.0);

	ConvolutionOperation conv(1, 1, 1, 1);
	TensorSet out(x.shape());

	conv.filter_bank.fill(1);

	ReluOperation t;
	TensorSet t_out(t.output_shape(out.shape()));
					
	conv.forward(x.x, out.x);
	t.forward(out.x, t_out.x);
	
	cout << x.x.to_vector() << endl;
	cout << out.x.to_vector() << endl;
	cout << t_out.x.to_vector() << endl;
	
	t_out.grad.fill(-1);
	t.backward(out.x, t_out.x, t_out.grad, out.grad);
	conv.backward(x.x, out.x, out.grad, x.grad);
	
	cout << x.grad.to_vector() << endl;
	cout << out.grad.to_vector() << endl;
	cout << t_out.grad.to_vector() << endl;

}

void test2() {
	DataBase db("/home/marijnfs/dev/caffe/examples/cifar10/cifar10_train_leveldb");
	DataBase db_test("/home/marijnfs/dev/caffe/examples/cifar10/cifar10_test_leveldb");
	Indices indices(db.N);

	// db.floatify();

	double std(.05);
	double lr(.001);


	//Network network(TensorShape{1, 10, 1, 1});
	//network.add_conv(10, 1, 1);

	
	Network network(TensorShape{1, 3, 4, 4});
	//Network network(TensorShape{1, 3, 32, 32});
	Tensor data(TensorShape{1, 3, 4, 4});
	data.init_normal(0, .5);
	vector<float> bla = data.to_vector();

	network.add_conv(5, 3, 3);
	network.add_pool(2, 2);
	
	network.add_squash(10);
	network.add_softmax();
	network.finish();

	network.init_normal(0, std);

	caffe::Datum datum = db.get_image(0);
	//const float *img_data = datum.float_data().data();
	const float *img_data = &bla[0];

	network.forward(img_data);
	network.calculate_loss(datum.label());
	network.backward();
	
	vector<float> grad = network.gradient();
	cout << grad.size() << endl;

	vector<float> fd_grad = network.fd_gradient(img_data, datum.label(), .0001);
	cout << fd_grad.size() << endl;
	for (size_t i(0); i < grad.size(); ++i)
		if (abs<float>(grad[i] + fd_grad[i]) > .001)
			cout << i << " " << grad[i] << " " << fd_grad[i] << " " << abs<float>(grad[i] + fd_grad[i]) << endl;
}

int main() {
	DataBase db("/home/marijnfs/dev/caffe/examples/cifar10/cifar10_train_leveldb");
	DataBase db_test("/home/marijnfs/dev/caffe/examples/cifar10/cifar10_test_leveldb");
	Indices indices(db.N);

	// db.floatify();

	double std(.05);
	double lr(.001);

	int n(1), c(3), h(32), w(32);
	int outc(10);
	Network network(TensorShape{n, c, w, h});
	network.add_conv(100, 3, 3);
	network.add_pool(2, 2);
	network.add_conv(100, 1, 1);
	network.add_relu();

	network.add_conv(200, 2, 2);
	network.add_pool(2, 2);
	network.add_conv(200, 1, 1);
	network.add_relu();

	network.add_conv(300, 2, 2);
	network.add_pool(2, 2);
	network.add_conv(300, 1, 1);
	network.add_relu();

	network.add_conv(400, 2, 2);
	network.add_pool(2, 2);
	network.add_conv(400, 1, 1);
	network.add_relu();

	network.add_conv(500, 2, 2);
	network.add_pool(2, 2);
	network.add_conv(500, 1, 1);
	network.add_relu();

	network.add_squash(10);
	network.add_softmax();
	network.finish();
	network.init_normal(0, std);

	//conv_operation1.init_normal(0, STD);
	//squash_operation.init_normal(0, STD);

	for (size_t e(0); e < 50; ++e) {
		Timer t;
		float err(0);
		int n_correct(0);

		indices.shuffle();
		for (size_t i(0); i < db.N; ++i) {
			//cout << i << endl;
			caffe::Datum datum = db.get_image(indices[i]);
			const float *img_data = datum.float_data().data();

			//Tensor t(1, 10, 1, 1);
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

