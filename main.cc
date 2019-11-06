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

void unet_test() {
    int in_channels(2);
    int out_channels(2);

	int next_c = 2;
	int k = 3;

	auto network = std::make_unique<Network<float>>();
	auto in = network->input_3D(in_channels);

	//in = network->local_normalisation_3D(9)(in);
    
	auto node = network->convolution_3D(next_c, k)(in);
	node = network->relu()(node);
	node = network->convolution_3D(next_c, k)(node);
	node = network->relu()(node);

	auto node_l1 = network->convolution_downscale_3D(next_c * 2, 2)(node);
	node_l1 = network->relu()(node_l1);
	node_l1 = network->convolution_3D(next_c * 2, k)(node_l1);
	//node_l1 = network->local_normalisation_3D(5)(node_l1);
	node_l1 = network->relu()(node_l1);

	auto node_l2 = network->convolution_downscale_3D(next_c * 4, 2)(node_l1);
	node_l2 = network->relu()(node_l2);
	node_l2 = network->convolution_3D(next_c * 4, k)(node_l2);
	//node_l2 = network->local_normalisation_3D(5)(node_l2);
	node_l2 = network->relu()(node_l2);

	auto node_l3 = network->convolution_downscale_3D(next_c * 8, 2)(node_l2);
	node_l3 = network->relu()(node_l3);
	node_l3 = network->convolution_3D(next_c * 8, k)(node_l3);
	//node_l3 = network->local_normalisation_3D(5)(node_l3);
	node_l3 = network->relu()(node_l3);

	node_l3 = network->convolution_upscale_3D(next_c * 4, 2)(node_l3);
	node_l2 = network->addition()(node_l2, node_l3);
	node_l2 = network->convolution_3D(next_c * 4, k)(node_l2);

	node_l2 = network->convolution_upscale_3D(next_c * 2, 2)(node_l2);
	node_l1 = network->addition()(node_l1, node_l2);
	node_l1 = network->convolution_3D(next_c * 2, k)(node_l1);

	node_l1 = network->convolution_upscale_3D(next_c, 2)(node_l1);
	node = network->addition()(node_l1, node);
	node = network->convolution_3D(next_c, k)(node);

	node = network->relu()(node);
	node = network->convolution_3D(out_channels, k)(node);
	auto prediction = network->sigmoid()(node);

	auto targetNode = network->input_3D(out_channels);
    auto loss = network->squared_loss()(prediction, targetNode);
    
    //init shapes
    Tensor<float> sample(TensorShape{1, in_channels, 64, 64, 64});
    Tensor<float> y(TensorShape{1, out_channels, 64, 64, 64});
	
    network->init_normal(0.0, 0.1);
    sample.init_normal(0.0, 0.1);
    y.init_normal(0.0, 0.1);

	network->save("test.dexe");
	loss({sample, y});
	cout << loss.tensor_set().x->to_vector() << endl;
	network->init_normal(0.0, 0.1);
	loss({sample, y});
	cout << loss.tensor_set().x->to_vector() << endl;

	network->load("test.dexe");
	loss({sample, y});
	cout << loss.tensor_set().x->to_vector() << endl;

    while (true) {
		loss({sample, y});
        network->zero_grad();
        loss.backward();
        network->update(0.01);
		cout << loss.tensor_set().x->to_vector() << endl;
    }
}


void test3() {

	cout << "start" << endl;
	Handler::set_device(0);
	Handler::cudnn();
	cout << "Test 3" << endl;

	{
		Network<double> net;


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
		int next_c = 8;
		int k = 3;

		// auto node = net.convolution_downscale_3D(next_c, 2)(in1);
		// auto prediction = net.convolution_upscale_3D(next_c, 2)(node);
		
		auto node = net.convolution_3D(next_c, k)(in1);
		auto node2 = net.convolution_3D(next_c, k)(in1);
		node = net.addition()(node, node2);		
		node = net.relu()(node);
		node = net.convolution_3D(next_c, k)(node);
		auto node1 = net.relu()(node);
		node = net.convolution_3D(next_c, k)(node1);
		node = net.relu()(node);
		node = net.addition()(node, node1);
		node = net.convolution_3D(next_c, k)(node);
		node = net.relu()(node);

		auto prediction = net.convolution_3D(1, k)(node);
		
		auto target = net.input_3D(in_c);
		auto loss = net.squared_loss()(prediction, target);

		//allocate input and target
		//remove need for this
		// in1.tensor_set().alloc_x(TensorShape{1, 1, 64, 64, 64});
		// target.tensor_set().alloc_x(TensorShape{1, 1, 64, 64, 64});

		Tensor<double> sample(TensorShape{1, 1, 64, 64, 64});
		Tensor<double> y(TensorShape{1, 1, 64, 64, 64});
		
		net.init_normal(0.0, 0.1);
		sample.init_normal(0.0, 0.1);
		y.init_normal(0.0, 0.1);

		loss({sample, y});

		// int l = 1;
		// cout << "next: " << net.names[l] << " " << net.tensors[l].x->to_vector() << endl;

		net.new_backward();
		auto grad = net.gradient();
		// y.from_tensor(sample);

		// auto fd_grad = net.fd_gradient(0.0000001);
		// cout << fd_grad.size() << " " << grad.size() << endl;
		// for (int n(0); n < fd_grad.size(); ++n) {
		// 	cout << "[" << fd_grad[n] << " " << grad[n] << " " << (fd_grad[n] / grad[n]) << "] ";
		// }

		for (int i(0); i < 4; ++i) {
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
    unet_test();
	//test3();
}

