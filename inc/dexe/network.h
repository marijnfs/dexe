#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <istream>

#include "dexe/config.h"
#include "dexe/util.h"
#include "dexe/tensor.h"
#include "dexe/cudavec.h"

namespace dexe {

struct Allocator; //predec

template <typename F>
struct Operation;

template <typename F>
struct Parametrised;

template <typename F>
struct Network;

template <typename F>
struct DEXE_API Node {
	Node(int index_ = -1, Network<F> *network_ = nullptr) : index(index_), network(network_) {}

	TensorShape shape();
	TensorSet<F> &tensor_set();

	void operator()(std::initializer_list<std::reference_wrapper<Tensor<F>>> inputs); //call to evaluation
	void backward() { network->backward(); }

	bool valid() { return index != -1; }

	void set_x(Tensor<F> &x);
	Tensor<F> &x();
    Tensor<F> &grad();

	int index = -1; //-1 means undefined
	Network<F> *network = nullptr;
    std::string name() { return network->names[index]; }
    std::string set_name(std::string name) { return network->names[index] = name; }
};

template <typename F>
struct DEXE_API Network {
	Network();
	~Network();

	Network(Network const &other) = delete;
	Network operator=(Network const &other) = delete;

	void reset();
	void empty_tensors(bool skip_input);
	
 	int add_operation(Operation<F> *op, std::vector<int> inputs, TensorShape shape, std::string name);

	std::function<Node<F>(Node<F>)> convolution_1D(int out_c, int k, int dilation, std::string name = "conv_1d");
	std::function<Node<F>(Node<F>)> convolution_1D(int out_c, int k, std::string name = "conv_1d");
	std::function<Node<F>(Node<F>)> convolution_2D(int out_c, int k, int dilation, std::string name = "conv_2d");
	std::function<Node<F>(Node<F>)> convolution_2D(int out_c, int k, std::string name = "conv_2d");
	std::function<Node<F>(Node<F>)> convolution_3D(int out_c, int k, int dilation, std::string name = "conv_3d");
	std::function<Node<F>(Node<F>)> convolution_3D(int out_c, int k, std::string name = "conv_3d");
	std::function<Node<F>(Node<F>)> convolution_downscale(int out_c, int k, std::string name = "downscale");
	std::function<Node<F>(Node<F>)> convolution_downscale_3D(int out_c, int k, std::string name = "downscale");
	std::function<Node<F>(Node<F>)> convolution_upscale(int out_c, int k, std::string name = "upscale");
	std::function<Node<F>(Node<F>)> convolution_upscale_3D(int out_c, int k, std::string name = "upscale");
	std::function<Node<F>(Node<F>)> relu(std::string name = "relu");
	std::function<Node<F>(Node<F>)> elu(std::string name = "elu");
	std::function<Node<F>(Node<F>)> tanh(std::string name = "tanh");
	std::function<Node<F>(Node<F>)> sigmoid(std::string name = "sigmoid");
	std::function<Node<F>(Node<F>)> local_normalisation(int k, std::string name = "lnc");
	std::function<Node<F>(Node<F>)> local_normalisation_3D(int k, std::string name = "lnc");
	std::function<Node<F>(Node<F>)> instance_normalisation(std::string name = "instance_norm");
	std::function<Node<F>(Node<F>, Node<F>)> squared_loss(std::string name = "squared_loss");
	std::function<Node<F>(Node<F>, Node<F>)> support_loss(F support, std::string name = "support_loss");
	std::function<Node<F>(Node<F>, Node<F>)> dice_loss(F smoothing = 0.0, std::string name = "dice_loss");

	std::function<Node<F>(Node<F>, Node<F>)> addition(std::string name = "addition");
	// std::function<Node<F>(Node<F>)> pool(std::string name = "pool");

	Node<F> input_1D(int n_channels, std::string name = "input");
	Node<F> input_2D(int n_channels, std::string name = "input");
	Node<F> input_3D(int n_channels, std::string name = "input");

	Node<F> get_node(std::string name);
	Node<F> get_node(int index);
	Node<F> get_last_node();

	std::vector<int> find_sequence(std::vector<int> inputs, std::vector<int> outputs);

	void forward(std::vector<int> inputs, std::vector<int> outputs);
	void backward();

	//memory efficient forward
	void forward_nograd(std::vector<int> inputs, std::vector<int> outputs);

	void set_inputs(std::initializer_list<std::reference_wrapper<Tensor<F>>> inputs);

	void zero_x();
	void zero_grad();

	void finish();
	void assert_finished();

	void update(F lr);
	void l2(F l);
	void init_normal(F mean, F std);
    void init_uniform(F var);

	void save(std::string path);
	void load(std::string path);

	void save(std::ostream &ostream);
	void load(std::istream &istream);

	void describe(std::ostream &out);

	void register_params();
	void align_params();

	std::vector<F> to_vector();
	void from_vector(std::vector<F> &vec);
	std::vector<F> fd_gradient(F e);
	std::vector<F> gradient();


	Tensor<F> *output();
	Tensor<F> *output_grad();
	TensorShape output_shape() { return tensors.back().shape(); }


	// Tensor<F> &input();
	// Tensor<F> *input_grad();

	std::string get_unique_name(std::string name);


	std::vector<int> sequence;

	std::vector<std::string> names;
	std::vector<std::unique_ptr<Operation<F>>> operations;
	std::vector<TensorSet<F>> tensors;
	std::vector<std::vector<int>> input_indices; //inputs to every node

	std::vector<int> inputs; //indices to nodes that function as inputs

	std::vector<Parametrised<F>*> parameters;

	CudaVec<F> param_vec, grad_vec;
	std::vector<CudaVec<F>*> param_ptrs, grad_ptrs;
	std::vector<CudaVec<F>*> fast_param_ptrs, fast_grad_ptrs;

	std::set<std::string> names_set;

	int n_params = 0;
	bool finished = false; //for now we keep it at true


	std::vector<TensorShape> output_shapes_cache; //cache variable to store output shapes after preparation
	std::unique_ptr<Allocator> fixed_allocator;
};

}
