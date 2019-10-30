#include "network.h"
#include "util.h"

#include <algorithm>
#include <iterator>
#include <fstream>
#include <sstream>
#include <stack>
#include <set>
#include <vector>

using namespace std;

template <typename F>
TensorShape Node<F>::shape() { 
	return network->tensors[index].shape();
}

template <typename F>
Network<F>::Network(TensorShape in) : n_params(0), finished(false) {
	tensors.emplace_back(in);
}
template <typename F>
void Network<F>::finish() {
	align_params();
}

template <typename F>
void Network<F>::assert_finished() {
	if (!finished)
		throw StringException("call network.finish() before using network");
}

template <typename F>
void Network<F>::zero_x() {
	for (auto &tensor : tensors)
		if (tensor.x)
			tensor.x->zero();
}

/*
template <typename F>
void Network<F>::backward() {
	assert_finished();
	for (int i(operations.size() - 1); i >= 0; --i) {
		operations[i]->backward(*tensors[i]->x, *tensors[i+1]->x, *tensors[i+1]->grad, *tensors[i]->grad);
		operations[i]->backward_weights(*tensors[i]->x, *tensors[i+1]->grad);
	}
}

template <typename F>
void Network<F>::backward_data() {
	for (int i(operations.size() - 1); i >= 0; --i)
		operations[i]->backward(*tensors[i]->x, *tensors[i+1]->x, *tensors[i+1]->grad, *tensors[i]->grad);
}
*/

template <typename F>
void Network<F>::update(F lr) {
	assert_finished();
	for (size_t i(0); i < parameters.size(); ++i)
		parameters[i]->update(lr);
}

template <typename F>
void Network<F>::l2(F l) {
	assert_finished();
	for (size_t i(0); i < parameters.size(); ++i)
		parameters[i]->l2(l);
}

template <typename F>
void Network<F>::init_normal(F mean, F std) {
	for (size_t i(0); i < parameters.size(); ++i)
		parameters[i]->init_normal(mean, std);
}

template <typename F>
void Network<F>::init_uniform(F var) {
	for (size_t i(0); i < parameters.size(); ++i)
		parameters[i]->init_uniform(var);
}

template <typename F>
void Network<F>::save(std::string path) {
	ofstream of(path, ios::binary);
	vector<F> data = to_vector();
	byte_write_vec(of, data);
}

template <typename F>
void Network<F>::load(std::string path) {
	ifstream in(path, ios::binary);
	vector<F> data = byte_read_vec<F>(in);
	from_vector(data);
}

template <typename F>
vector<F> Network<F>::to_vector() {
	vector<F> full_vec;
	for (size_t i(0); i < parameters.size(); ++i) {
		vector<F> vec = parameters[i]->to_vector();
		copy(vec.begin(), vec.end(), back_inserter(full_vec));
	}
	return full_vec;
}

template <typename F>
void Network<F>::from_vector(vector<F> &vec) {
	cout << "in from vector" << endl;
	typename vector<F>::iterator it(vec.begin());
	for (size_t i(0); i < parameters.size(); ++i) {
		vector<F> v(it, it + parameters[i]->size());
		parameters[i]->from_vector(v);
		it += parameters[i]->size();
	}
	cout << "done from vector" << endl;
}

// template <typename F>
// vector<F> Network<F>::fd_gradient(F const *cpu_data, int label, F e) {
// 	vector<F> full_grad;

// 	for (size_t i(0); i < parameters.size(); ++i) {
// 		cout << "params: " << i << "/" << parameters.size() << endl;
// 		vector<F> vec = parameters[i]->to_vector();

// 		vector<F> delta_vec(vec);
// 		for (size_t n(0); n < vec.size(); ++n) {
// 			delta_vec[n] = vec[n] + e;
// 			parameters[i]->from_vector(delta_vec);

// 			forward(cpu_data);
// 			calculate_loss(label);
// 			F plus_loss = loss();

// 			delta_vec[n] = vec[n] - e;
// 			parameters[i]->from_vector(delta_vec);

// 			//throw "";
// 			forward(cpu_data);
// 			calculate_loss(label);
// 			F min_loss = loss();
// 			//cout << "+" << plus_loss << " " << min_loss << endl;

// 			full_grad.push_back((plus_loss - min_loss) / (2 * e));
// 			delta_vec[n] = vec[n];
// 		}
// 		parameters[i]->from_vector(vec);
// 	}
// 	return full_grad;
// }

template <typename F>
vector<F> Network<F>::gradient() {
	vector<F> full_grad;
	for (size_t i(0); i < parameters.size(); ++i) {
		vector<F> grad = parameters[i]->grad_to_vector();
		copy(grad.begin(), grad.end(), back_inserter(full_grad));
	}
	return full_grad;
}

// template <typename F>
// Tensor<F> &Network<F>::input() {
// 	assert_finished();
// 	return tensors[0]->x;
// }


template <typename F>
void Network<F>::describe(ostream &out) {
	for (auto &o : operations) {
		o->describe(out);
		out << endl;
	}
	out.flush();
}

template <typename F>
void Network<F>::align_params() {
	register_params();
	param_vec.resize(n_params);
	grad_vec.resize(n_params);

	for (auto &p : param_ptrs)
		cudaFree(*(p.ptr));

	for (auto &g : grad_ptrs)
		cudaFree(*(g.ptr));

	position_params(param_vec.data, grad_vec.data);
	cout << "n params: " << n_params << endl;
	//throw "";
}

template <typename F>
void Network<F>::register_params() {
	for (auto &p : parameters)
		p->register_params(param_ptrs, fast_param_ptrs, grad_ptrs, fast_grad_ptrs);

 	n_params = 0;
	for (auto &p : param_ptrs)
		n_params += p.n;
}

template <typename F>
void Network<F>::position_params(F *pos_param, F *pos_grad) {
	F *ptr = pos_param;
	for (auto &p : param_ptrs) {
		*(p.ptr) = ptr;
		ptr += p.n;
	}

	// ptr = pos_fast_param;
	// for (auto &p : fast_params) {
	// 	*(p.ptr) = ptr;
	// 	ptr += p.n;
	// }

	ptr = pos_grad;
	for (auto &g : grad_ptrs) {
		*(g.ptr) = ptr;
		ptr += g.n;
	}

	// ptr = pos_fast_grad;
	// for (auto &g : fast_grads) {
	// 	*(g.ptr) = ptr;
	// 	ptr += g.n;
	// }
}


template <typename F>
Network<F>::~Network() {
	cout << "DESTRUCTOR" << endl;
}

template <typename F>
string Network<F>::get_unique_name(string name) {
	int n(0);
	while (true) {
		ostringstream oss;
		oss << name << "_" << n;
		if (!names_set.count(oss.str())) 
			return oss.str();
		++n;
	}
	throw std::runtime_error("unreachable");
}

template <typename F>
int Network<F>::add_operation(Operation<F> *op, vector<int> inputs, TensorShape shape, string name) {
	int index = names.size();

	auto unique_name = get_unique_name(name);

	names.emplace_back(unique_name);
	names_set.insert(unique_name);

	operations.emplace_back(op);
	cout << "add operation, add tensor by shape " << shape << endl;
	tensors.emplace_back(shape);
	cout << "done add operation, add tensor by shape " << shape << endl;

	input_indices.emplace_back(inputs);
	cout << "added: " << name << " N:" << names.size() << endl;
	if (auto param = dynamic_cast<Parametrised<F>*>(op))
		parameters.emplace_back(param);
	return index;
}

template <typename F>
Node<F> Network<F>::input(int n_channels, std::string name) {
	auto index = add_operation(new InputOperation<F>(n_channels), vector<int>{}, TensorShape{0, n_channels, 0, 0}, name);
	return Node<F>(index, this);
}

template <typename F>
Node<F> Network<F>::input_3D(int n_channels, std::string name) {
	auto index = add_operation(new InputOperation<F>(n_channels), vector<int>{}, TensorShape{0, n_channels, 0, 0, 0}, name);
	return Node<F>(index, this);
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::convolution(int out_c, int k, string name) {
	return [this, out_c, k, name](Node<F> n) {
		auto in_c = n.shape().c();
		cout << "in_c : " << in_c << endl;
		
		auto index = add_operation(new ConvolutionOperation<F>({out_c, in_c, k, k}, {1, 1}, true), vector<int>{n.index}, TensorShape{0, out_c, 0, 0}, name);
		return Node<F>(index, this);
	};
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::convolution_3D(int out_c, int k, string name) {
	return [this, out_c, k, name](Node<F> n) {
		auto in_c = n.shape().c();
		cout << "in_c : " << in_c << endl;
		auto index = add_operation(new ConvolutionOperation<F>({out_c, in_c, k, k, k}, {1, 1, 1}, true), vector<int>{n.index}, TensorShape{0, out_c, 0, 0, 0}, name);
		return Node<F>(index, this);
	};
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::convolution_downscale(int out_c, int k, string name) {
	return [this, out_c, k, name](Node<F> n) {
		auto in_c = n.shape().c();
		auto index = add_operation(new ConvolutionOperation<F>({out_c, in_c, k, k}, {k, k}, false), vector<int>{n.index}, TensorShape{0, out_c, 0, 0}, name);
		return Node<F>(index, this);
	};
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::convolution_downscale_3D(int out_c, int k, string name) {
	return [this, out_c, k, name](Node<F> n) {
		auto in_c = n.shape().c();
		auto index = add_operation(new ConvolutionOperation<F>({out_c, in_c, k, k, k}, {k, k, k}, false), vector<int>{n.index}, TensorShape{0, out_c, 0, 0, 0}, name);
		return Node<F>(index, this);
	};
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::convolution_upscale(int out_c, int k, string name) {
	return [this, out_c, k, name](Node<F> n) {
		auto in_c = n.shape().c();
        // in Conv Transpose, the in_c and out_c ordering logic is reversed
		auto index = add_operation(new ConvolutionTransposeOperation<F>({in_c, out_c, k, k}, {k, k}, false), vector<int>{n.index}, TensorShape{0, out_c, 0, 0}, name);
		return Node<F>(index, this);
	};
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::convolution_upscale_3D(int out_c, int k, string name) {
	return [this, out_c, k, name](Node<F> n) {
		auto in_c = n.shape().c();
        // in Conv Transpose, the in_c and out_c ordering logic is reversed
		auto index = add_operation(new ConvolutionTransposeOperation<F>({in_c, out_c, k, k, k}, {k, k, k}, false), vector<int>{n.index}, TensorShape{0, out_c, 0, 0, 0}, name);
		return Node<F>(index, this);
	};
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::relu(string name) {
	return [this, name](Node<F> n) {
		auto in_c = n.shape().c();
		auto index = add_operation(new ReluOperation<F>(), vector<int>{n.index}, TensorShape{0, in_c, 0, 0}, name);

		return Node<F>(index, this);
	};
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::local_normalisation(int k, string name) {
	return [this, name, k](Node<F> n) {
		auto in_c = n.shape().c();
		auto index = add_operation(new LocalNormalisationOperation<F>(k), vector<int>{n.index}, TensorShape{0, in_c, 0, 0}, name);

		return Node<F>(index, this);
	};
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::local_normalisation_3D(int k, string name) {
	return [this, name, k](Node<F> n) {
		auto in_c = n.shape().c();
		auto index = add_operation(new LocalNormalisationOperation<F>(k), vector<int>{n.index}, TensorShape{0, in_c, 0, 0, 0}, name);

		return Node<F>(index, this);
	};
}

template <typename F>
std::function<Node<F>(Node<F>, Node<F>)> Network<F>::addition(string name) {
	return [this, name](Node<F> n1, Node<F> n2) {
		cout << n1.index << " " << n2.index << endl;
		cout << n1.shape() << ", " << n2.shape() << endl;
		
		auto in_c = n1.shape().c();
		auto index = add_operation(new AdditionOperation<F>(), vector<int>{n1.index, n2.index}, TensorShape{0, in_c, 0, 0}, name);

		return Node<F>(index, this);
	};
}


template <typename F>
void Network<F>::new_forward(std::vector<int> inputs, std::vector<int> outputs) {
	vector<int> sequence;
	set<int> input_set(inputs.begin(), inputs.end());

	for (auto i : inputs)
		if (!tensors[i].x) {
			cerr << "Input tensor at index " << i << " is not set" << endl;
			return;
		}

	//Build the forward dependency (resolution) graph
	vector<vector<int>> output_indices(operations.size());
	for (int n(0); n < operations.size(); ++n)
		for (int in_n : input_indices[n])
			output_indices[in_n].push_back(n);

	stack<int> node_stack;
	set<int> resolved_nodes;

	for (auto i : inputs)
		node_stack.push(i);

	while (!node_stack.empty()) {
		auto cur = node_stack.top();
		bool resolved = true;
		for (auto n : output_indices[cur])
			if (!resolved_nodes.count(n)) {
				node_stack.push(n);
				resolved = false;
			}
		
		if (!resolved)
			continue;

		node_stack.pop();
		resolved_nodes.insert(cur);
		cout << endl << "cur :" << cur << " " << names[cur] << endl;

		//don't add leaf nodes to calculation sequence
		sequence.push_back(cur);
	}

	//put calculation nodes in order
	reverse(sequence.begin(), sequence.end());

	//Forward Dryrun
	cout << "DryRun " << sequence << endl;
	for (auto s : sequence) {
		vector<Tensor<F>*> tmp_inputs, tmp_outputs;
		cout << "seq: " << s << "op: " << names[s] << endl;
		for (auto idx : input_indices[s]) {
			cout << "idx: " << idx << endl;
			cout << "input: " << names[idx] << endl;
			cout << "in: " << tensors[idx].x->shape << endl;
			tmp_inputs.push_back(tensors[idx].x.get());
		}

		tmp_outputs.push_back(tensors[s].x.get());
		bool success = operations[s]->forward_dry_run(tmp_inputs, tmp_outputs);
		if (!success) {
			ostringstream oss;
			oss << "Failure when preparing step: " << names[s] << endl;
			throw std::runtime_error(oss.str());
		}

		//make sure x is zero
		if (!input_set.count(s))
			tensors[s].x->zero();
	}

	//Run Forward
	cout << "Actual Run" << endl;
	for (auto s : sequence) {
		cout << "running layer: " << names[s] << endl;
		vector<Tensor<F>*> tmp_inputs, tmp_outputs;
		for (auto idx : input_indices[s])
			tmp_inputs.push_back(tensors[idx].x.get());
		tmp_outputs.push_back(tensors[s].x.get());

		operations[s]->forward(tmp_inputs, tmp_outputs);
	}
}


template struct Network<float>;
// template struct Network<double>;
