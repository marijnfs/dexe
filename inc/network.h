#ifndef __NETWORK_H__
#define __NETWORK_H__


#include <iostream>
#include <vector>
#include <set>
#include <functional>

#include "util.h"
#include "tensor.h"
#include "operations.h"
#include "cudavec.h"

template <typename F>
struct Network;

template <typename F>
struct Node {
	Node(int index_, Network<F> *network_) : index(index_), network(network_) {}

	TensorShape shape();

	int index = -1; //-1 means undefined
	Network<F> *network = nullptr;
};

template <typename F>
struct Network {
	Network(){}
	Network(TensorShape in);
	~Network();

 	int add_operation(Operation<F> *op, std::vector<int> inputs, TensorShape shape, std::string name);

	std::function<Node<F>(Node<F>)> convolution(int out_c, int k, std::string name = "conv");
	std::function<Node<F>(Node<F>)> convolution_3D(int out_c, int k, std::string name = "conv");
	std::function<Node<F>(Node<F>)> convolution_downscale(int out_c, int k, std::string name = "downscale");
	std::function<Node<F>(Node<F>)> convolution_downscale_3D(int out_c, int k, std::string name = "downscale");
	std::function<Node<F>(Node<F>)> convolution_upscale(int out_c, int k, std::string name = "upscale");
	std::function<Node<F>(Node<F>)> convolution_upscale_3D(int out_c, int k, std::string name = "upscale");
	std::function<Node<F>(Node<F>)> relu(std::string name = "relu");
	std::function<Node<F>(Node<F>, Node<F>)> addition(std::string name = "addition");
	// std::function<Node<F>(Node<F>)> pool(std::string name = "pool");

	Node<F> input(int n_channels, std::string name = "input");
	Node<F> input_3D(int n_channels, std::string name = "input");

	void new_forward(std::vector<int> inputs, std::vector<int> outputs);


	void finish();
	void assert_finished();

	//void forward();
	//void forward(F const *cpu_data);

	//void backward(F const *cpu_data);
	//void backward();
	//void backward_data();

	void update(F lr);
	void l2(F l);
	void init_normal(F mean, F std);
    void init_uniform(F var);

	void save(std::string path);
	void load(std::string path);

	void describe(std::ostream &out);

	void register_params();
	void align_params();
	void position_params(F *pos_param, F *pos_grad);

	std::vector<F> to_vector();
	void from_vector(std::vector<F> &vec);
	// std::vector<F> fd_gradient(F const *cpu_data, int label, F e);
	std::vector<F> gradient();


	Tensor<F> *output();
	Tensor<F> *output_grad();
	TensorShape output_shape() { return tensors.back().shape(); }


	// Tensor<F> &input();
	// Tensor<F> *input_grad();

	std::string get_unique_name(std::string name);


	std::vector<std::string> names;
	std::vector<std::unique_ptr<Operation<F>>> operations;
	std::vector<TensorSet<F>> tensors;
	std::vector<std::vector<int>> input_indices;

	std::vector<Parametrised<F>*> parameters;

	CudaVec param_vec, grad_vec;
	std::vector<CudaPtr<F>> param_ptrs, grad_ptrs;
	std::vector<CudaPtr<F>> fast_param_ptrs, fast_grad_ptrs;

	std::set<std::string> names_set;

	int n_params;
	bool finished;
};

#endif
