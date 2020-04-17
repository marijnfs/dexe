#include <iostream>
#include <stdint.h>
#include "dexe/operations.h"
#include "dexe/util.h"
#include "dexe/network.h"
#include "dexe/colour.h"
#include "dexe/optimizer.h"
#include "dexe/io.h"
#include "dexe/models.h"
//#include <unistd.h>
#include <ctime>
#include <cuda.h>

using namespace std;
using namespace dexe;

void unet_test(string path) {
    auto network = make_unique<Network<double>>();
    int in_channels = 1;
    int out_channels = 1;
    
	auto target = network->input_3D(out_channels);
    auto prediction = make_unet(network.get(), in_channels, out_channels);

	float support = 0.5;
    auto loss = network->support_loss(support)(prediction, target);

    //init shapes
    Tensor<double> sample(TensorShape{1, in_channels, 8, 8, 8});
    Tensor<double> y(TensorShape{1, out_channels, 8, 8, 8});
	
    network->init_uniform(0.05);
    sample.init_normal(0.0, 0.1);
    y.init_normal(0.0, 0.1);
    y.threshold(0.0);
    sample.from_tensor(y);

	//SGDOptimizer<double> optimizer(0.01);
	//AdaOptimizer<double> optimizer(0.01, 0.95);
    AdamOptimizer<double> optimizer(0.01, 0.95, 0.99);
	optimizer.register_network(*network);

	int epoch(0);
    while (true) {
		loss({sample, y});
        network->zero_grad();
        loss.backward();
        optimizer.update();
		cout << loss.tensor_set().x->to_vector() << endl;
		if (epoch++ > 1000)
			break;
    }
    cout << prediction.tensor_set().x->to_vector() << endl;
    cout << y.to_vector() << endl;
    cout << sample.to_vector() << endl;
}

void new_test() {
	auto network = make_unique<Network<float>>();

	auto input = network->input_3D(1);
	network->convolution_3D(4, 3)(input);
	
}

void test3() {

	cout << "start" << endl;
	Handler::set_device(0);
	Handler::cudnn();
	cout << "Test 3" << endl;

	for (int n(0); n < 2; ++n)
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
		node = net.instance_normalisation()(node);
		node = net.addition()(node, node2);
        auto prediction = net.convolution_3D(1, k)(node);
		
		auto target = net.input_3D(in_c);
		//auto loss = net.squared_loss()(prediction, target);
        auto loss = net.support_loss(0.4)(prediction, target);
        
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

        // cout << "predict " << prediction.tensor_set().x->to_vector() << endl;
		// int l = 1;
		// cout << "next: " << net.names[l] << " " << net.tensors[l].x->to_vector() << endl;

		net.backward();
		auto grad = net.gradient();

		// y.from_tensor(sample);

        cout << "start fd grad" << endl;
        net.empty_tensors(true);
        net.set_inputs({sample, y});
        auto fd_grad = net.fd_gradient(0.0000001);
        // cout << fd_grad.size() << " " << grad.size() << endl;
        for (int n(0); n < fd_grad.size(); ++n) {
		 	cout << "[" << fd_grad[n] << " " << grad[n] << " " << (fd_grad[n] / grad[n]) << "] " << endl;
        }
        
/*		for (int i(0); i < 4; ++i) {
			cout << "it: " << i << endl;
			loss({sample, y});
			net.zero_grad();

			cout << "loss vec: " << loss.tensor_set().x->to_vector() << endl;
			loss.backward();
			net.update(0.01);
		}
*/

	}
    cout << "deinit" << endl;
    Handler::deinit();

	cout << "Done" << endl;
}



void test4() {

	cout << "start" << endl;
	Handler::set_device(0);
	Handler::cudnn();
	cout << "Test 3" << endl;

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
	// node = net.instance_normalisation()(node);
	//auto node2 = net.convolution_3D(next_c, k)(in1);
	//node = net.addition()(node, node2);		
    auto prediction = net.convolution_3D(1, k)(node);
	
	// auto target = net.input_3D(in_c);
	//auto loss = net.squared_loss()(prediction, target);
    // auto loss = net.support_loss(0.4)(prediction, target);
    
	//allocate input and target
	//remove need for this
	// in1.tensor_set().alloc_x(TensorShape{1, 1, 64, 64, 64});
	// target.tensor_set().alloc_x(TensorShape{1, 1, 64, 64, 64});

	Tensor<double> sample(TensorShape{1, 1, 64, 64, 64});
	Tensor<double> y(TensorShape{1, 1, 64, 64, 64});
	
	net.init_normal(0.0, 0.1);
	sample.init_normal(0.0, 0.1);
	y.init_normal(0.0, 0.1);

    net.set_inputs({sample});
    net.forward_nograd(net.inputs, {prediction.index});
}

int load_and_run(int argc, char **argv) {
	if (argc < 2) {
		throw std::runtime_error("need argument");
	}

	string filename(argv[1]);

	Network<float> net;
	net.load(filename);

	auto out_node = net.get_node("output");
	if (!out_node.valid())
	{
		throw std::runtime_error("No output node available");
	}

	Tensor<float> sample(TensorShape{1, 1, 64, 64, 64});
	net.set_inputs({sample});
	net.forward_nograd(net.inputs, {out_node.index});
}

int main(int argc, char **argv) {
	// Tensor<double> bla(TensorShape({1, 1, 4}));
	// bla.init_normal(0.0, 4.0);
	// cout << bla.norm2() << endl;
	// bla /= sqrt(bla.norm2() / bla.size());
	// cout << bla.norm2() << endl;
	// return 0;
    // if (argc < 2)
    //     throw std::runtime_error("need argument");
    //unet_test(argv[1]);
	//test3();
	return load_and_run(argc, argv);
}

