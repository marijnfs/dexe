#pragma once

#include "network.h"
#include "cudavec.h"
#include "config.h"

namespace dexe {

template <typename F>
struct DEXE_API Optimizer {
    Optimizer();
    virtual ~Optimizer();

    virtual void register_network(Network<F> &network);
    virtual void update();
};

template <typename F>
struct DEXE_API SGDOptimizer : public Optimizer<F> {
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
struct DEXE_API AdaOptimizer : public Optimizer<F> {
    AdaOptimizer(F lr_, F beta_ = 0.95);
   	~AdaOptimizer() = default;

    virtual void register_network(Network<F> &network);
    virtual void update();

    void set_lr(F lr);
    
    Network<F> *network = nullptr;
    CudaVec<F> std;
    CudaVec<F> tmp, tmp2;
    F lr = 0;
	F beta = 0.0;
    F eps = 0.0001;
};


template <typename F>
struct DEXE_API AdamOptimizer : public Optimizer<F> {
    AdamOptimizer(F lr_, F beta_ = 0.9, F momentum_factor_ = 0.95);
   	~AdamOptimizer() = default;

    virtual void register_network(Network<F> &network);
    virtual void update();

    void set_lr(F lr);
    void set_momentum(F momentum);
    
    Network<F> *network = nullptr;
    CudaVec<F> momentum, std;
    CudaVec<F> tmp, tmp2;
    F lr = 0;
	F beta = 0.0;
	F momentum_factor = 0.0;
    F eps = 0.0001;
};

}
