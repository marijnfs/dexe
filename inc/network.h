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
struct Node {
	Node(int index_, Operation<F> *op_) : index(index_), op(op_) {}

	int index = -1; //-1 means undefined
	Operation<F> *op;
};

template <typename F>
struct Network {
	Network(){}
	Network(TensorShape in);
	~Network();


	std::function<Node<F>(Node<F>)> convolution(int k, std::string name = "conv");
	Node<F> input(std::string name = "input");


	void add_conv(int outmap, int kw, int kh);
	void add_pool(int kw, int kh);
	void add_squash(int outmap);
	void add_tanh();
	void add_relu();
	void add_softmax();

  void add_unsquash(TensorShape shape);
  void add_merge();
  void add_split();
  
	void add_operation(Operation<F> *op);
	void finish();
	void assert_finished();

	void forward();
	void forward(F const *cpu_data);

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
	void position_params(F *pos_param, F *pos_grad);

	std::vector<F> to_vector();
	void from_vector(std::vector<F> &vec);
	// std::vector<F> fd_gradient(F const *cpu_data, int label, F e);
	std::vector<F> gradient();


	Tensor<F> *output();
	Tensor<F> *output_grad();
  TensorShape output_shape() { return last(shapes); }

  void add_operation(Operation<F> *op, std::vector<int> inputs, std::string name);

	// Tensor<F> &input();
	Tensor<F> *input_grad();

	std::string get_unique_name(std::string name);


	std::vector<std::string> names;
	std::vector<std::unique_ptr<Operation<F>>> operations;
	std::vector<std::unique_ptr<TensorSet<F>>> tensors;
	std::vector<std::vector<int>> input_indices;

	std::vector<Parametrised<F>*> parameters;
	std::vector<TensorShape> shapes;

	CudaVec param_vec, grad_vec;
	std::vector<CudaPtr<F>> param_ptrs, grad_ptrs;
	std::vector<CudaPtr<F>> fast_param_ptrs, fast_grad_ptrs;

	std::set<std::string> names_set;

	int n_params;
	bool finished;
};

#endif
