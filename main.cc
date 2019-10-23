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
	auto in1 = net.input(in_c);
	int next_c = 4;
	int k = 3;
	auto node2 = net.convolution(next_c, k)(in1);
	auto added = net.addition()(in1, node2);


	net.new_forward(vector<int>{0}, vector<int>{2});
	cout << "Done" << endl;
}

int main() {
	test3();
}

