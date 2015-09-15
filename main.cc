#include <cudnn.h>
#include <iostream>
#include <stdint.h>
#include "operations.h"
#include "util.h"
#include "database.h"
#include "network.h"
#include "img.h"
#include "colour.h"
#include "adv.h"
#include "balancer.h"

#include <unistd.h>
#include <ctime>
#include <cuda.h>

using namespace std;

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
	test1();
	return 1;
	Balancer::start(2);

	srand(time(0));
	Database<caffe::Datum> db("/home/marijnfs/dev/caffe-rk/examples/cifar10/cifar10_train_leveldb");
	Database<caffe::Datum> db_test("/home/marijnfs/dev/caffe-rk/examples/cifar10/cifar10_test_leveldb");
	Database<caffe::Datum> db_adv("./adv");
	db_adv.from_database(db); //copy

	//db.normalize_chw();
	//db_test.normalize_chw();

	double std(.05);
	double lr(.001);

	int n(1), c(3), h(32), w(32);
	int outc(10);
	Network<float> network(TensorShape{n, c, w, h});	

	/*
	network.add_conv(128, 8, 8);
	network.add_pool(4, 4);
	network.add_tanh();
	network.add_conv(196, 4, 4);
	network.add_pool(2, 2);
	network.add_tanh();
	network.add_conv(196, 2, 2);
	network.add_pool(2, 2);
	network.add_tanh();
	*/

	network.add_conv(100, 3, 3);
	network.add_pool(2, 2);
	network.add_tanh();
	
	network.add_conv(200, 3, 3);
	network.add_pool(2, 2);
	network.add_tanh();
	
	network.add_conv(300, 3, 3);
	network.add_pool(2, 2);
	//network.add_conv(300, 1, 1);
	network.add_tanh();

	network.add_conv(400, 3, 3);
	network.add_pool(2, 2);
	//network.add_conv(400, 1, 1);
	network.add_tanh();

	network.add_conv(500, 3, 3);
	network.add_pool(2, 2);
	//network.add_conv(500, 1, 1);
	network.add_tanh();

	//fully connected
	//network.add_squash(100);
	//network.add_tanh();

	/*
	network.add_conv(96, 9, 9);
	network.add_pool(2, 2);
	network.add_tanh();

	network.add_conv(100, 9, 9);
	network.add_pool(2, 2);
	network.add_tanh();

	network.add_conv(100, 5, 5);
	network.add_pool(2, 2);
	network.add_tanh();

	*/

	network.add_squash(100);
	network.add_tanh();

	network.add_squash(10);
	network.add_softmax();
	network.finish();
	//network.init_normal(0, std);
	network.init_uniform(std);

	//conv_operation1.init_normal(0, STD);
	//squash_operation.init_normal(0, STD);

	Balancer::init(2);
	Balancer::advance(1, 300*3);

	for (size_t e(0); e < 50000; ++e) {
		//if ((e % 4 == 0))
		//MakeAdvDatabase(db, db_adv, network, 15.);
		
		//MakeAdvDatabase(db, db_adv, network, 15.);
		//if (e > 6 && (e % 2 == 0)) {

		/*
		if (Balancer::ready(0)) {
			//int n(db.N);
			int n(1000);
			Balancer::start(0);
			AddNAdv(db, db_adv, network, n, 5., 5);
			Balancer::stop(0);
			}*/

		cout << "epoch: " << e << " lr: " << lr << endl;
		lr *= .99;
		Timer t;
		float err(0);
		int n_correct(0);
		
		Indices indices(db_adv.N);
		indices.shuffle();

		size_t i(0);
		for (size_t i(0); i < db_adv.N; ++i) {
		//while (Balancer::ready(1)) {
			Balancer::start(1);
			//cout << i << endl;
			//caffe::Datum datum = db.get_image(49999);
			//for (size_t n(0); n < datum.float_data_size(); ++n)
			//		cout << datum.float_data(n) << " ";
			//cout << endl;

			//cout << "float data size: " << datum.float_data_size() << endl;
			
			//caffe::Datum datum = db.get_image(indices[i]);
			caffe::Datum datum = db_adv.get_image(indices[i]);
			const float *img_data = datum.float_data().data();


			// =================
			// Normal backprop
			network.forward(img_data);
			network.calculate_loss(datum.label());

			if (network.loss() > .01) {  //speedup, skip if prob > .99
				network.backward();
				network.update(lr);
			}

			/*
			// ================
			// Adversarial backprop
			network.forward(img_data);
			network.calculate_loss(datum.label());
			network.backward_data();

			vector<float> data_grad = network.tensors[0]->grad.to_vector();
			//float grad_norm = norm(data_grad);
			float grad_norm = l1_norm(data_grad);
			float factor = rand_float();
			if (i % 5000 == 0)
				network.tensors[0]->x.write_img("norm.bmp");
		   	//add_cuda<float>(network.tensors[0]->grad.data, network.tensors[0]->x.data, network.tensors[0]->x.size(), -1 * factor / (grad_norm * grad_norm));
			cout << "cost1: " << network.loss() << endl;
			add_cuda<float>(network.tensors[0]->grad.data, network.tensors[0]->x.data, network.tensors[0]->x.size(), -.3 * datum.float_data().size() / (grad_norm));

			if (i % 5000 == 0) {
				cout << "factor: " << factor << endl;
				network.tensors[0]->x.write_img("adv.bmp");
			}
			*/
			/*
			//little test
			{
				network.forward(img_data);
				for (size_t i(0); i < 1000; ++i) {
					network.forward();
					network.calculate_loss(datum.label());
					network.backward_data();
					add_cuda<float>(network.tensors[0]->grad.data, network.tensors[0]->x.data, network.tensors[0]->x.size(), -.3 * datum.float_data().size() / (grad_norm) / 1000.);
				}
				network.tensors[0]->x.write_img("adv_smooth.bmp");
				network.forward();
				network.calculate_loss(datum.label());
				cout << "cost3: " << network.loss() << endl;

				MakeAdvDatabase(db, db_adv, network, .3);
				throw "";
			}
			*/
			/*
			network.forward();
			network.calculate_loss(datum.label());
			network.backward();
			network.update(lr);
			*/			
			// ========================


			//network.l2(.0001);


			err += network.loss();
			n_correct += network.n_correct();
			
			if (i % 5000 == 0) {
				vector<float> ov = network.output().to_vector();
				cout << i << " (" << indices[i] << ") " << datum.label() << " " << ov << " " << ov[datum.label()] << endl;
			}
			i = (i + 1) % db_adv.N;
			Balancer::stop(1);
		}


		float test_err(0);
		int test_n_correct(0);
		
		for (size_t i(0); i < db_test.N; ++i) {
			caffe::Datum datum = db_test.get_image(i);
			const float *img_data = datum.float_data().data();

			network.forward(img_data);
			network.calculate_loss(datum.label());
				
			test_err += network.loss();
			test_n_correct += network.n_correct();
		}

		cout << "elapsed: " << t.since() << " err: " << RED << (err / db_adv.N) << DEFAULT << " correct: " << BLUE << n_correct << DEFAULT << "/" << db_adv.N << endl;
		cout << "test err: " << RED << (test_err / db_test.N) << DEFAULT << " correct: " << BLUE << test_n_correct << DEFAULT << "/" << db_test.N << endl;
	}
}

