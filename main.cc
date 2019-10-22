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

	auto in1 = net.input();
	auto node2 = net.convolution(3)(in1);
	cout << "Done" << endl;
}

int main() {
	test3();
}

