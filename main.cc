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
		auto c1 = net.convolution_3D(next_c, k)(in1);
		auto c1f = net.relu()(c1);
		auto c2 = net.convolution_3D(1, k)(c1f);
		
		auto target = net.input_3D(in_c);
		auto loss = net.squared_loss()(c2, target);

		in1.tensor_set().alloc_x(TensorShape{1, 1, 64, 64, 64});
		target.tensor_set().alloc_x(TensorShape{1, 1, 64, 64, 64});

		Tensor<float> sample(TensorShape{1, 1, 64, 64, 64});

		loss({sample});
		net.new_forward(vector<int>{in1.index, target.index}, vector<int>{loss.index});


	}
    cout << "deinit" << endl;
    Handler::deinit();

	cout << "Done" << endl;
}

int main() {
	test3();
}

