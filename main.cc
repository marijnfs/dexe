#include <cudnn.h>
#include <iostream>
#include <stdint.h>
#include "operations.h"
#include "util.h"
#include "network.h"
#include "img.h"
#include "colour.h"

#include <unistd.h>
#include <ctime>
#include <cuda.h>

using namespace std;

void test3() {

	cout << "start" << endl;
	Handler::set_device(0);
	Handler::cudnn();
	cout << "Test 3" << endl;

	{
		Network<float> net;


		// int in_c = 1;
		// auto in1 = net.input3D(in_c);
		// int next_c = 4;
		// int k = 3;
		// auto node2 = net.convolution3D(next_c, k)(in1);
		// // auto node2f = net.relu()(node2);
		// auto node3 = net.convolution3D(1, k)(node2);

		int in_c = 1;
		auto in1 = net.input_3D(in_c);
		// auto in2 = net.input3D(in_c);
		int next_c = 16;
		int k = 3;
		auto node = net.convolution_3D(next_c, k)(in1);
		node = net.relu()(node);
		node = net.convolution_3D(next_c, k)(node);
		auto node1 = net.relu()(node);
		node = net.convolution_3D(next_c, k)(node1);
		node = net.relu()(node);
		node = net.addition()(node, node1);
		node = net.convolution_3D(next_c, k)(node);
		node = net.relu()(node);

		auto c2 = net.convolution_3D(1, k)(node);
		
		auto target = net.input_3D(in_c);
		auto loss = net.squared_loss()(c2, target);

		//allocate input and target
		//remove need for this
		// in1.tensor_set().alloc_x(TensorShape{1, 1, 64, 64, 64});
		// target.tensor_set().alloc_x(TensorShape{1, 1, 64, 64, 64});

		Tensor<float> sample(TensorShape{1, 1, 64, 64, 64});
		Tensor<float> y(TensorShape{1, 1, 64, 64, 64});
		
		net.init_normal(0.0, 0.05);
		sample.init_normal(0.0, 1.0);
		y.init_normal(0.0, 1.0);
		y.from_tensor(sample);

		for (int i(0); i < 100000; ++i) {
			cout << "it: " << i << endl;
			loss({sample, y});
			net.zero_grad();

			cout << "loss vec: " << loss.tensor_set().x->to_vector() << endl;
			loss.backward();
			net.update(0.01);
		}


	}
    cout << "deinit" << endl;
    Handler::deinit();

	cout << "Done" << endl;
}

int main() {
	test3();
}

