#ifndef __NETWORK_H__
#define __NETWORK_H__


#include "util.h"
#include "tensor.h"
#include "operations.h"
#include "loss.h"

template <typename F>
struct Network {
	Network(TensorShape in);
	~Network();

	void add_conv(int outmap, int kw, int kh);
	void add_pool(int kw, int kh);
	void add_squash(int outmap);
	void add_tanh();
	void add_relu();
	void add_softmax();

	void add_operation(Operation<F> *op);
	void finish();
	void assert_finished();

	void forward();
	void forward(F const *cpu_data);
	void calculate_loss(int label);
	void calculate_loss(std::vector<int> &labels);
	void calculate_loss(Tensor<F> &target);
	void calculate_average_loss();
	void backward(F const *cpu_data);
	void backward();
	void backward_data();

	void update(F lr);
	void l2(F l);
	void init_normal(F mean, F std);
    void init_uniform(F var);

	std::vector<F> to_vector();
	std::vector<F> fd_gradient(F const *cpu_data, int label, F e);
	std::vector<F> gradient();


	Tensor<F> &output();
	Tensor<F> &input();
	F loss();
	F n_correct();

	std::vector<Parametrised<F>*> params;
	std::vector<Operation<F>*> operations;
	std::vector<TensorSet<F>*> tensors;
	std::vector<TensorShape> shapes;

	Loss<F> *loss_ptr;
	bool finished;
};

#endif
