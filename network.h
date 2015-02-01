#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "operations.h"
#include "util.h"
#include "tensor.h"


struct Network {
	Network(TensorShape in);
	~Network();

	void add_conv(int outmap, int kw, int kh);
	void add_pool(int kw, int kh);
	void add_squash(int outmap);
	void add_tanh();
	void add_relu();
	void add_softmax();

	void add_operation(Operation *op);
	void finish();
	void assert_finished();

	void forward(float const *cpu_data);
	void calculate_loss(int label);
	void calculate_loss(std::vector<int> &labels);
	void backward();

	void update(float lr);
	void l2(float l);
	void init_normal(float mean, float std);

	std::vector<float> to_vector();
	std::vector<float> fd_gradient(float const *cpu_data, int label, float e);
	std::vector<float> gradient();



	Tensor &output();
	float loss();
	float n_correct();

	std::vector<Parametrised*> params;
	std::vector<Operation*> operations;
	std::vector<TensorSet*> tensors;
	std::vector<TensorShape> shapes;

	SoftmaxLoss *loss_ptr;
	bool finished;
};

#endif
