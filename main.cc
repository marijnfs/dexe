#include <cudnn.h>
#include <iostream>
#include <stdint.h>
#include "operations.h"
#include "util.h"
#include "database.h"
#include "network.h"
#include "img.h"
#include "colour.h"

#include <unistd.h>
#include <ctime>
#include <cuda.h>

using namespace std;

void test3() {
	cout << "Test 3" << endl;

	Network<float> net;

	int in_c = 1;
	auto in1 = net.input3D(in_c);
	auto in2 = net.input3D(in_c);

	// int next_c = 4;
	// int k = 3;
	// auto node2 = net.convolution3D(next_c, k)(in1);
	auto node2 = net.addition()(in1, in2);

	cout << "adding addition" << endl;

	cout << "before: " << net.tensors[0].shape() << endl;
	net.tensors[0].alloc_x(TensorShape{1, 1, 64, 64, 64});
	net.tensors[1].alloc_x(TensorShape{1, 1, 64, 64, 64});
	cout << "after: " << net.tensors[0].shape() << endl;
	
	net.new_forward(vector<int>{0, 1}, vector<int>{node2.index});
	cout << "Done" << endl;
}

int main() {
	test3();
}

