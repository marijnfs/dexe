#include <cudnn.h>
#include <iostream>
#include <stdint.h>
#include "operations.h"
#include "util.h"
#include "database.h"
#include "network.h"

using namespace std;

int main() {
	DataBase db("/home/marijnfs/dev/caffe/examples/cifar10/cifar10_train_leveldb");
	vector<int> indices;
	for (size_t i(0); i < db.N; ++i) indices[i] = i;
	// db.floatify();

	double std(.005);
	double lr(.005);

	int n(1), c1(3), c2(50), h(32), w(32);
	int outc(10);

	Network network(TensorShape{n, c1, w, h});
	network.add_conv(c2, 5, 5);
	network.add_tanh();
	network.add_pool(4, 4);
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

		random_shuffle(indices);
		for (size_t i(0); i < db.N; ++i) {
			//cout << i << endl;
			caffe::Datum datum = db.get_image(indices[i]);

			const float *img_data = datum.float_data().data();

			network.forward(img_data);
			network.backward(datum.label());
			network.update(lr);

			err += network.loss();
			n_correct += network.n_correct();
			
			if (i % 5000 == 0) {
				vector<float> ov = network.output().to_vector();
				cout << i << " " << datum.label() << " " << ov << " " << ov[datum.label()] << endl;
			}
		}
		cout << "elapsed: " << t.since() << " err: " << (err / db.N) << " correct: " << n_correct << "/" << db.N << endl;
	}
}

