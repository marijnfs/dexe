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

	double STD(.005);

	int n(1), c1(3), c2(50), h(32), w(32);
	int outc(10);

	Tensor h1(n, c1, h, w);
	Tensor h2(n, c2, h, w);
	Tensor h3(n, c2, h, w);

	Tensor o1(n, outc, 1, 1); 
	Tensor o(n, outc, 1, 1);


	Tensor g_h1(n, c1, h, w);
	Tensor g_h2(n, c2, h, w);
	Tensor g_h3(n, c2, h, w);

	Tensor g_o1(n, outc, 1, 1); 


	ConvolutionLayer conv_layer1(c1, c2, 5, 5);
	TanhLayer tanh_layer;
	SquashLayer squash_layer(h3, 10);
	SoftmaxLayer softmax;
	SoftmaxLossLayer softmax_loss(n, outc);

	conv_layer1.init_normal(0, STD);
	squash_layer.init_normal(0, STD);

	for (size_t e(0); e < 10; ++e) {
		Timer t;
		float err(0);

		for (size_t i(0); i < db.N; ++i) {
			caffe::Datum datum = db.get_image(i);


			const float *img_data = datum.float_data().data();

			h1.from_ptr(img_data);
			
			conv_layer1.forward(h1, h2);
			tanh_layer.forward(h2, h3);
			squash_layer.forward(h3, o1);
			softmax.forward(o1, o);

			// cout << o.to_vector() << endl;
			softmax_loss.forward(o, datum.label());
			softmax.backward(o1, softmax_loss.err, g_o1);
			
			squash_layer.backward_weights(h3, g_o1);
			squash_layer.backward_input(g_o1, g_h3);
			// cout << squash_layer.filter_bank_grad.to_vector() << endl;
			// return -1;
			
			tanh_layer.backward(h2, h3, g_h3, g_h2);
			conv_layer1.backward_weights(h1, g_h2);

			conv_layer1.update(.0002);
			squash_layer.update(.0002);

			err += softmax_loss.loss();
			if (i % 5000 == 0) cout << i << " " << datum.label() << " " << o.to_vector() << endl;
			// conv_layer1.backward_input(g_h2, g_h1);

			// cout << h2.to_vector() << endl;
			// vector<float> vec2 = t2_map.to_vector();
			// vec2.resize(10);
			// cout << "t2out" << vec2 << endl << vec2 << endl;

			// Tensor out(n, w2, h2, 2);
		}
		cout << "elapsed: " << t.since() << " err: " << (err / db.N) << endl;

	}
}

