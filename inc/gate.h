#ifndef __GATE_H__
#define __GATE_H__

__global__ void gate_kernel(int N, float const *a, float const *b, float *out);

template <typename F>
void gate(Tensor<F> &a, Tensor<F> &b, Tensor<F> &out);

#endif
