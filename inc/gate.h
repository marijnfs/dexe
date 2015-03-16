#ifndef __GATE_H__
#define __GATE_H__

#include "tensor.h"
__global__ void gate_kerneld(int N, double const *a, double const *b, double *out);
__global__ void gate_kernelf(int N, float const *a, float const *b, float *out);

template <typename F>
void gate(Tensor<F> &a, Tensor<F> &b, Tensor<F> &out);

#endif
