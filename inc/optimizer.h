#pragma once

#include "network.h"
#include "cudavec.h"

template <typename F>
struct Optimizer {
    Optimizer();
    virtual ~Optimizer();

    virtual void register_network(Network<F> &network);
    virtual void update();

    
};

template <typename F>
struct SGDOptimizer : public Optimizer<F> {
    SGDOptimizer(F lr_);
   	~SGDOptimizer() = default;

    virtual void register_network(Network<F> &network);
    virtual void update();

    void set_lr(F lr);
    
    Network<F> *network = nullptr;
    CudaVec<F> tmp;
    F lr = 0;
};

template <typename F>
struct AdaOptimizer : public Optimizer<F> {
    AdaOptimizer(F lr_);
   	~AdaOptimizer() = default;

    virtual void register_network(Network<F> &network);
    virtual void update();

    void set_lr(F lr);
    
    Network<F> *network = nullptr;
    CudaVec<F> std;
    CudaVec<F> tmp, tmp2;
    F lr = 0;
	F beta = 0.9;
    F eps = 0.01;
};
