#pragma once

#include "network.h"

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

    virtual void register_network(Network<F> &network);
    virtual void update();

    void set_lr(F lr);
    
    F lr = 0;
};
