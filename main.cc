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
		int next_c = 64;
		int k = 3;
		auto c1 = net.convolution_3D(next_c, k)(in1);
		auto c1f = net.relu()(c1);
		auto c2 = net.convolution_3D(next_c, k)(c1f);
		auto c2f = net.relu()(c2);
		auto c3 = net.convolution_3D(next_c, k)(c2f);
		auto c3f = net.relu()(c3);
		auto c4 = net.convolution_3D(next_c, k)(c3f);
		auto c4f = net.relu()(c4);
		auto c4_t = net.convolution_transpose_3D(next_c / 2, 2)(c3f);
			// auto node2 = net.addition()(in1, in2);

		cout << "adding addition" << endl;

		cout << "before: " << net.tensors[0].shape() << endl;
		net.tensors[0].alloc_x(TensorShape{1, 1, 64, 64, 64});
		// net.tensors[1].alloc_x(TensorShape{1, 1, 64, 64, 64});
		cout << "after: " << net.tensors[0].shape() << endl;

		cout << "running forward:" << endl;
		net.new_forward(vector<int>{0}, vector<int>{c4_t.index});
        cout << net.tensors[c4_t.index].x->shape << endl;
	}
    cout << "deinit" << endl;
    Handler::deinit();

	cout << "Done" << endl;
}

int main() {
	test3();
}

