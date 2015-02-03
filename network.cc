#include "network.h"
#include <algorithm>
#include <iterator>

using namespace std;

Network::Network(TensorShape in) : loss_ptr(0), finished(false) {
	shapes.push_back(in);
	tensors.push_back(new TensorSet(in));
}

void Network::add_conv(int outmap, int kw, int kh) {
	ConvolutionOperation *conv = new ConvolutionOperation(last(shapes).c, outmap, kw, kh);
	add_operation(conv);
	params.push_back(conv);
}

void Network::add_pool(int kw, int kh) {
	add_operation(new PoolingOperation(kw, kh));
}

void Network::add_squash(int c) {
	SquashOperation *squash = new SquashOperation(last(shapes), c);
	add_operation(squash);
	params.push_back(squash);
}

void Network::add_tanh() {
	add_operation(new TanhOperation());
}

void Network::add_relu() {
	add_operation(new ReluOperation());
}

void Network::add_softmax() {
	add_operation(new SoftmaxOperation());
}

void Network::add_operation(Operation *op) {
	operations.push_back(op);
	shapes.push_back(last(operations)->output_shape(last(shapes)));
	tensors.push_back(new TensorSet(last(shapes)));
}

void Network::finish() {
	loss_ptr = new SoftmaxLoss(last(shapes).n, last(shapes).c);
	//loss_ptr = new SquaredLoss(last(shapes).n, last(shapes).c);
	finished = true;
}

void Network::assert_finished() {
	if (!finished)
		throw StringException("call network.finish() before using network");
}

void Network::forward(float const *cpu_data) {
	assert_finished();
	first(tensors)->x.from_ptr(cpu_data);
	
	for (size_t i(0); i < operations.size(); ++i)
		operations[i]->forward(tensors[i]->x, tensors[i+1]->x);
}

void Network::calculate_loss(int label) {
	assert_finished();
	loss_ptr->calculate_loss(last(tensors)->x, label, last(tensors)->grad);
}

void Network::calculate_loss(std::vector<int> &labels) {
	assert_finished();
	loss_ptr->calculate_loss(last(tensors)->x, labels, last(tensors)->grad);
}

void Network::backward() {
	for (int i(operations.size() - 1); i >= 0; --i) {
		operations[i]->backward(tensors[i]->x, tensors[i+1]->x, tensors[i+1]->grad, tensors[i]->grad);
		operations[i]->backward_weights(tensors[i]->x, tensors[i+1]->grad);
	}
}

void Network::update(float lr) {
	assert_finished();
	for (size_t i(0); i < params.size(); ++i)
		params[i]->update(lr);
}

void Network::l2(float l) {
	assert_finished();
	for (size_t i(0); i < params.size(); ++i)
		params[i]->l2(l);
}

void Network::init_normal(float mean, float std) {
	for (size_t i(0); i < params.size(); ++i)
		params[i]->init_normal(mean, std);
}

vector<float> Network::to_vector() {
	vector<float> full_vec;
	for (size_t i(0); i < params.size(); ++i) {
		vector<float> vec = params[i]->to_vector();
		copy(vec.begin(), vec.end(), back_inserter(full_vec));
	}
	return full_vec;
}

vector<float> Network::fd_gradient(float const *cpu_data, int label, float e) {
	vector<float> full_grad;

	for (size_t i(0); i < params.size(); ++i) {
		cout << "params: " << i << "/" << params.size() << endl;
		vector<float> vec = params[i]->to_vector();
		vector<float> delta_vec(vec);
		for (size_t n(0); n < vec.size(); ++n) {
			delta_vec[n] = vec[n] + e;
			params[i]->from_vector(delta_vec);

			forward(cpu_data);
			calculate_loss(label);
			float plus_loss = loss();

			delta_vec[n] = vec[n] - e;
			params[i]->from_vector(delta_vec);

			//throw "";
			forward(cpu_data);
			calculate_loss(label);
			float min_loss = loss();
			//cout << "+" << plus_loss << " " << min_loss << endl;
			
			full_grad.push_back((plus_loss - min_loss) / (2 * e));
			delta_vec[n] = vec[n];
		}
		params[i]->from_vector(vec);
	}
	return full_grad;
}

vector<float> Network::gradient() {
	vector<float> full_grad;
	for (size_t i(0); i < params.size(); ++i) {
		vector<float> grad = params[i]->grad_to_vector();
		copy(grad.begin(), grad.end(), back_inserter(full_grad));
	}
	return full_grad;
}

Tensor &Network::output() {
	assert_finished();
	return last(tensors)->x;
}

float Network::loss() {
	assert_finished();
	return loss_ptr->loss();
}

float Network::n_correct() {
	assert_finished();
	return loss_ptr->n_correct();
}

Network::~Network() {
	del_vec(operations);
	del_vec(tensors);
	delete loss_ptr;
}
