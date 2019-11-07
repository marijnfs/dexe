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

template <typename F>
SGDOptimizer<F>::SGDOptimizer(F lr_) : lr(lr_) {
}

template <typename F>
void SGDOptimizer<F>::register_network(Network<F> &network) {
	
}

template <typename F>
void SGDOptimizer<F>::update() {
	
}

template <typename F>
void SGDOptimizer<F>::set_lr(F lr) {
}
