#include <cudnn.h>
#include <iostream>
#include <stdint.h>
#include "layers.h"
#include "util.h"
using namespace std;

int main() {
	double STD(.005);


	int n(1), c(2), h(480), w(640);
	
	Tensor t1(n, w, h, c);
	t1.init_normal(0, .5);

	int c2(50), h2(480), w2(640);
	Tensor t2(n, w2, h2, c2), t2_act(n, w2, h2, c2);
	Tensor t2_map(n, w2, h2, c2), t2_out(n, w2, h2, c2);
	Tensor t2_bias(1, 1, 1, c2);
	t2_bias.init_normal(0, 0);

	Tensor t3(n, w2, h2, c2), t3_act(n, w2, h2, c2);
	Tensor t3_map(n, w2, h2, c2), t3_out(n, w2, h2, c2);
	Tensor t3_bias(1, 1, 1, c2);
	t3_bias.init_normal(0, 0);

	Tensor t4(n, w2, h2, c2), t4_act(n, w2, h2, c2);
	Tensor t4_map(n, w2, h2, c2), t4_out(n, w2, h2, c2);
	Tensor t4_bias(1, 1, 1, c2);
	t4_bias.init_normal(0, 0);

	Tensor fc1(n, w2, h2, c2), fc1_tan(n, w2, h2, c2), fc2(n, w2, h2, 2),fc2_soft(n, w2, h2, 2);

	//create filters


	ConvolutionLayer conv_layer1(c, c2, 4, 4);
	conv_layer1.init_normal(0, STD);

	ConvolutionLayer conv_layer2(c2, c2, 4, 4);
	conv_layer2.init_normal(0, STD);

	ConvolutionLayer conv_layer3(c2, c2, 2, 2);
	conv_layer3.init_normal(0, STD);


	TanhLayer tanh_layer;
	SoftmaxLayer softmax;

	Timer t;

	conv_layer1.forward(t1, t2);
	

  tanh_layer.forward(t2, t2_act);
  
  vector<float> vec2 = t2_map.to_vector();
  vec2.resize(10);
  cout << "t2out" << vec2 << endl << vec2 << endl;
  
  
  conv_layer2.forward(t2_out, t3);
  tanh_layer.forward(t3, t3_act);
  
  conv_layer3.forward(t3_out, t4);
  tanh_layer.forward(t4, t4_act);
  
  Tensor out(n, w2, h2, 2);
	cout << "elapsed: " << t.since() << endl;
}

