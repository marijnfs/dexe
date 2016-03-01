#include "network.h"
#include "util.h"

#include <algorithm>
#include <iterator>
#include <fstream>

using namespace std;

template <typename F>
Network<F>::Network(TensorShape in) : loss_ptr(0), n_params(0), finished(false) {
	shapes.push_back(in);
	tensors.push_back(new TensorSet<F>(in));
}

template <typename F>
void Network<F>::add_conv(int outmap, int kw, int kh) {
	ConvolutionOperation<F> *conv = new ConvolutionOperation<F>(last(shapes).c, outmap, kw, kh, true, 512 * 1024 * 1024);
	add_operation(conv);
	params.push_back(conv);
}

template <typename F>
void Network<F>::add_pool(int kw, int kh) {
	add_operation(new PoolingOperation<F>(kw, kh));
}

template <typename F>
void Network<F>::add_squash(int c) {
	SquashOperation<F> *squash = new SquashOperation<F>(last(shapes), c);
	add_operation(squash);
	params.push_back(squash);
}

template <typename F>
void Network<F>::add_tanh() {
	add_operation(new TanhOperation<F>());
}

template <typename F>
void Network<F>::add_relu() {
	add_operation(new ReluOperation<F>());
}

template <typename F>
void Network<F>::add_softmax() {
	add_operation(new SoftmaxOperation<F>());
}

template <typename F>
void Network<F>::add_operation(Operation<F> *op) {
	operations.push_back(op);
	shapes.push_back(last(operations)->output_shape(last(shapes)));
	tensors.push_back(new TensorSet<F>(last(shapes)));
}

template <typename F>
void Network<F>::finish() {
	loss_ptr = new SoftmaxLoss<F>(last(shapes).n, last(shapes).c);
	//loss_ptr = new SquaredLoss<F>(last(shapes).n, last(shapes).c);

	for (size_t i(0); i < operations.size(); ++i)
		operations[i]->forward_dry_run(tensors[i]->x, tensors[i+1]->x);

	finished = true;

	align_params();
}

template <typename F>
void Network<F>::assert_finished() {
	if (!finished)
		throw StringException("call network.finish() before using network");
}

template <typename F>
void Network<F>::forward(F const *cpu_data) {
	assert_finished();
	first(tensors)->x.from_ptr(cpu_data);

	forward();
}

template <typename F>
void Network<F>::forward() {
	assert_finished();

	for (size_t i(0); i < operations.size(); ++i)
		operations[i]->forward(tensors[i]->x, tensors[i+1]->x);
}

template <typename F>
void Network<F>::calculate_loss(int label) {
	assert_finished();
	loss_ptr->calculate_loss(last(tensors)->x, label, last(tensors)->grad);
}

template <typename F>
void Network<F>::calculate_loss(std::vector<int> &labels) {
	assert_finished();
	loss_ptr->calculate_loss(last(tensors)->x, labels, last(tensors)->grad);
}

template <typename F>
void Network<F>::calculate_average_loss() {
	assert_finished();
	loss_ptr->calculate_average_loss(last(tensors)->x, last(tensors)->grad);
}

template <typename F>
void Network<F>::calculate_loss(Tensor<F> &target) {
	assert_finished();
	loss_ptr->calculate_loss(last(tensors)->x, target, last(tensors)->grad);
}

template <typename F>
void Network<F>::backward(F const * cpu_data) {
	assert_finished();
	last(tensors)->grad.from_ptr(cpu_data);

	backward();
}


template <typename F>
void Network<F>::backward() {
	assert_finished();
	for (int i(operations.size() - 1); i >= 0; --i) {
		operations[i]->backward(tensors[i]->x, tensors[i+1]->x, tensors[i+1]->grad, tensors[i]->grad);
		operations[i]->backward_weights(tensors[i]->x, tensors[i+1]->grad);
	}
}

template <typename F>
void Network<F>::backward_data() {
	for (int i(operations.size() - 1); i >= 0; --i)
		operations[i]->backward(tensors[i]->x, tensors[i+1]->x, tensors[i+1]->grad, tensors[i]->grad);
}

template <typename F>
void Network<F>::update(F lr) {
	assert_finished();
	for (size_t i(0); i < params.size(); ++i)
		params[i]->update(lr);
}

template <typename F>
void Network<F>::l2(F l) {
	assert_finished();
	for (size_t i(0); i < params.size(); ++i)
		params[i]->l2(l);
}

template <typename F>
void Network<F>::init_normal(F mean, F std) {
	for (size_t i(0); i < params.size(); ++i)
		params[i]->init_normal(mean, std);
}

template <typename F>
void Network<F>::init_uniform(F var) {
	for (size_t i(0); i < params.size(); ++i)
		params[i]->init_uniform(var);
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
	for (size_t i(0); i < params.size(); ++i) {
		vector<F> vec = params[i]->to_vector();
		copy(vec.begin(), vec.end(), back_inserter(full_vec));
	}
	return full_vec;
}

template <typename F>
void Network<F>::from_vector(vector<F> &vec) {
	cout << "in from vector" << endl;
	typename vector<F>::iterator it(vec.begin());
	for (size_t i(0); i < params.size(); ++i) {
		vector<F> v(it, it + params[i]->size());
		params[i]->from_vector(v);
		it += params[i]->size();
	}
	cout << "done from vector" << endl;
}

template <typename F>
vector<F> Network<F>::fd_gradient(F const *cpu_data, int label, F e) {
	vector<F> full_grad;

	for (size_t i(0); i < params.size(); ++i) {
		cout << "params: " << i << "/" << params.size() << endl;
		vector<F> vec = params[i]->to_vector();

		vector<F> delta_vec(vec);
		for (size_t n(0); n < vec.size(); ++n) {
			delta_vec[n] = vec[n] + e;
			params[i]->from_vector(delta_vec);

			forward(cpu_data);
			calculate_loss(label);
			F plus_loss = loss();

			delta_vec[n] = vec[n] - e;
			params[i]->from_vector(delta_vec);

			//throw "";
			forward(cpu_data);
			calculate_loss(label);
			F min_loss = loss();
			//cout << "+" << plus_loss << " " << min_loss << endl;

			full_grad.push_back((plus_loss - min_loss) / (2 * e));
			delta_vec[n] = vec[n];
		}
		params[i]->from_vector(vec);
	}
	return full_grad;
}

template <typename F>
vector<F> Network<F>::gradient() {
	vector<F> full_grad;
	for (size_t i(0); i < params.size(); ++i) {
		vector<F> grad = params[i]->grad_to_vector();
		copy(grad.begin(), grad.end(), back_inserter(full_grad));
	}
	return full_grad;
}

template <typename F>
Tensor<F> &Network<F>::output() {
	assert_finished();
	return last(tensors)->x;
}

template <typename F>
Tensor<F> &Network<F>::output_grad() {
	assert_finished();
	return last(tensors)->grad;
}

template <typename F>
Tensor<F> &Network<F>::input() {
	assert_finished();
	return tensors[0]->x;
}

template <typename F>
Tensor<F> &Network<F>::input_grad() {
	assert_finished();
	return tensors[0]->grad;
}

template <typename F>
F Network<F>::loss() {
	assert_finished();
	return loss_ptr->loss();
}

template <typename F>
F Network<F>::n_correct() {
	assert_finished();
	return loss_ptr->n_correct();
}

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
	for (auto &o : params)
		o->register_params(param_ptrs, fast_param_ptrs, grad_ptrs, fast_grad_ptrs);

 	n_params = 0;
	for (auto &p : param_ptrs)
		n_params += p.n;
}

template <typename F>
void Network<F>::position_params(float *pos_param, float *pos_grad) {
	float *ptr = pos_param;
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
	del_vec(operations);
	del_vec(tensors);
	delete loss_ptr;
}

template struct Network<float>;
// template struct Network<double>;
