#ifndef __NETWORK_H__
#define __NETWORK_H__


#include <iostream>
#include <vector>
#include "util.h"
#include "tensor.h"
#include "operations.h"
#include "loss.h"
#include "cudavec.h"

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

	void save(std::string path);
	void load(std::string path);

	void describe(std::ostream &out);

	void register_params();
	void align_params();
	void position_params(float *pos_param, float *pos_grad);

	std::vector<F> to_vector();
	void from_vector(std::vector<F> &vec);
	std::vector<F> fd_gradient(F const *cpu_data, int label, F e);
	std::vector<F> gradient();


	Tensor<F> &output();
	Tensor<F> &output_grad();

	Tensor<F> &input();
	Tensor<F> &input_grad();

	F loss();
	F n_correct();

	std::vector<Parametrised<F>*> params;
	std::vector<Operation<F>*> operations;
	std::vector<TensorSet<F>*> tensors;
	std::vector<TensorShape> shapes;

	CudaVec param_vec, grad_vec;
	std::vector<CudaPtr<F>> param_ptrs, grad_ptrs;
	std::vector<CudaPtr<F>> fast_param_ptrs, fast_grad_ptrs;

	Loss<F> *loss_ptr;
	int n_params;
	bool finished;
};

#endif
