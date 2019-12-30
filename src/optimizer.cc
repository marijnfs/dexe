#include "dexe/optimizer.h"

using namespace std;

namespace dexe {

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
AdaOptimizer<F>::AdaOptimizer(F lr_, F beta_) : lr(lr_), beta(beta_) {
}

template <typename F>
void AdaOptimizer<F>::register_network(Network<F> &network_) {
	network = &network_;
	network->finish();

	std.resize(network->param_vec.N);
	std += 0.1;
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


////////////////
template <typename F>
AdamOptimizer<F>::AdamOptimizer(F lr_, F beta_, F momentum_factor_) : lr(lr_), beta(beta_), momentum_factor(momentum_factor_) {
}

template <typename F>
void AdamOptimizer<F>::register_network(Network<F> &network_) {
	network = &network_;
	network->finish();

	momentum.resize(network->param_vec.N);
	std.resize(network->param_vec.N);
	std += 0.1;

	tmp.resize(network->param_vec.N);
	tmp2.resize(network->param_vec.N);
}

template <typename F>
void AdamOptimizer<F>::update() {
	if (!network)
		throw std::runtime_error("No network registered");
	tmp = network->grad_vec;
	tmp.pow(2);
	tmp *= (1.0 - beta);
	std *= beta;
	std += tmp;

	tmp2 = std;
	tmp2.sqrt();
	tmp2 += eps;
	tmp = network->grad_vec;
	tmp /= tmp2;

	momentum *= momentum_factor;
	tmp *= (1.0 - momentum_factor);
	momentum += tmp;
	tmp = momentum;
	tmp *= lr;

	network->param_vec += tmp;
}

template <typename F>
void AdamOptimizer<F>::set_lr(F lr_) {
	lr = lr_;
}
template struct Optimizer<float>;
template struct Optimizer<double>;
template struct SGDOptimizer<float>;
template struct SGDOptimizer<double>;
template struct AdaOptimizer<float>;
template struct AdaOptimizer<double>;
template struct AdamOptimizer<float>;
template struct AdamOptimizer<double>;

}
