#include "optimizer.h"

using namespace std;

template <typename F>
Optimizer<F>::Optimizer() {
}

template <typename F>
Optimizer<F>::~Optimizer() {
}

template <typename F>
void Optimizer<F>::register_network(Network<F> &network) {
}

template <typename F>
void Optimizer<F>::update() {
}

/////////////////
template <typename F>
SGDOptimizer<F>::SGDOptimizer(F lr_) : lr(lr_) {
}

template <typename F>
void SGDOptimizer<F>::register_network(Network<F> &network_) {
	network = &network_;
	network->finish();

	tmp.resize(network->param_vec.N);
}

template <typename F>
void SGDOptimizer<F>::update() {
	if (!network)
		throw std::runtime_error("No network registered");
	tmp = network->grad_vec;
	tmp *= lr;
	network->param_vec += tmp;
}

template <typename F>
void SGDOptimizer<F>::set_lr(F lr_) {
	lr = lr_;
}

////////////////
template <typename F>
AdaOptimizer<F>::AdaOptimizer(F lr_) : lr(lr_) {
}

template <typename F>
void AdaOptimizer<F>::register_network(Network<F> &network_) {
	network = &network_;
	network->finish();

	std.resize(network->param_vec.N);
	tmp.resize(network->param_vec.N);
	tmp2.resize(network->param_vec.N);
}

template <typename F>
void AdaOptimizer<F>::update() {
	if (!network)
		throw std::runtime_error("No network registered");
	tmp = network->grad_vec;
	tmp.pow(2);
	tmp *= (1.0 - beta);
	std *= beta;
	std += tmp;

	tmp = network->grad_vec;
	tmp2 = std;
	tmp2.sqrt();
	tmp2 += eps;

	tmp /= tmp2;
	tmp *= lr;
	network->param_vec += tmp;
}

template <typename F>
void AdaOptimizer<F>::set_lr(F lr_) {
	lr = lr_;
}

template struct Optimizer<float>;
template struct Optimizer<double>;
template struct SGDOptimizer<float>;
template struct SGDOptimizer<double>;
template struct AdaOptimizer<float>;
template struct AdaOptimizer<double>;
