#include <iostream>
#include <stdint.h>
#include "dexe/operations.h"
#include "dexe/util.h"
#include "dexe/network.h"
#include "dexe/img.h"
#include "dexe/colour.h"
#include "dexe/optimizer.h"
#include "dexe/io.h"
//#include <unistd.h>
#include <ctime>
#include <cuda.h>

using namespace std;
using namespace dexe;

template <typename F>
function<Node<F>(Node<F>)> basic_layer(int c, int c_out, int k) {
    return [c, c_out, k](Node<F> node) {
               auto network = node.network;
               if (!network)
                   throw std::runtime_error("basic_layer: No network in node");
               auto node_last = network->convolution_3D(c_out, k)(node);
               node = network->convolution_3D(c, k)(node);
               node = network->relu()(node);
               node = network->convolution_3D(c_out, k)(node);
               node = network->addition()(node_last, node);
               

               //node = network->relu()(node);  
               return node;
           };
}

template <typename F>
Node<F> make_unet(Network<F> *network, int in_channels, int out_channels) {
	int c = 2;
	int k = 3;
    int norm_k = 5;

	auto in = network->input_3D(in_channels);
    auto in_normalised = network->local_normalisation_3D(norm_k)(in);

    auto l0 = basic_layer<F>(c, c, k)(in_normalised);

    // ^
    int c1 = c << 1;
    auto l1 = network->convolution_downscale_3D(c1, 2)(l0);
    l1 = basic_layer<F>(c1, c1, k)(l1);

    // ^
    int c2 = c << 2;
    auto l2 = network->convolution_downscale_3D(c2, 2)(l1);
    l2 = basic_layer<F>(c2, c2, k)(l2);

    // ^
    int c3 = c << 3;
	auto l3 = network->convolution_downscale_3D(c3, 2)(l2);
	l3 = basic_layer<F>(c3, c3, k)(l3);

    // 
    auto l2_down = network->convolution_upscale_3D(c2, 2)(l3);
    l2_down = basic_layer<F>(c2, c2, k)(l2_down);
    l2 = network->addition()(l2, l2_down);

    auto l1_down = network->convolution_upscale_3D(c1, 2)(l2_down);
    l1_down = basic_layer<F>(c1, c1, k)(l1_down);
    l1 = network->addition()(l1, l1_down);

    auto l0_down = network->convolution_upscale_3D(c, 2)(l1_down);
    l0_down = basic_layer<F>(c, c, k)(l0_down);
    l0 = network->addition()(l0, l0_down);

    auto prediction = network->convolution_3D(out_channels, k)(l0);    
    return prediction;
}

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
		node = net.instance_normalisation()(node);
		//auto node2 = net.convolution_3D(next_c, k)(in1);
		//node = net.addition()(node, node2);		
        auto prediction = net.convolution_3D(1, k)(node);
		
		auto target = net.input_3D(in_c);
		//auto loss = net.squared_loss()(prediction, target);
        auto loss = net.support_loss(0.4)(prediction, target);
        
		//allocate input and target
		//remove need for this
		// in1.tensor_set().alloc_x(TensorShape{1, 1, 64, 64, 64});
		// target.tensor_set().alloc_x(TensorShape{1, 1, 64, 64, 64});

		Tensor<double> sample(TensorShape{1, 1, 4, 4, 4});
		Tensor<double> y(TensorShape{1, 1, 4, 4, 4});
		
		net.init_normal(0.0, 0.1);
		sample.init_normal(0.0, 0.1);
		y.init_normal(0.0, 0.1);

		loss({sample, y});

        cout << "predict " << prediction.tensor_set().x->to_vector() << endl;
		// int l = 1;
		// cout << "next: " << net.names[l] << " " << net.tensors[l].x->to_vector() << endl;

		net.new_backward();
		auto grad = net.gradient();
		// y.from_tensor(sample);

        cout << "start fd grad" << endl;
        auto fd_grad = net.fd_gradient(0.0000001);	
        cout << fd_grad.size() << " " << grad.size() << endl;
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
	test3();
}

