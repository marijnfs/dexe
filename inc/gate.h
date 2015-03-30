#ifndef __GATE_H__
#define __GATE_H__

#include "tensor.h"
__global__ void gate_kernelf(int N, float const *a, float const *b, float *out);
__global__ void gate_kerneld(int N, double const *a, double const *b, double *out);

template <typename F>
void gate(Tensor<F> &a, Tensor<F> &b, Tensor<F> &out);

__global__ void range_kernelf(float *a, int N, float const min, float const max);
__global__ void range_kerneld(double *a, int N, double const min, double const max);

template <typename F>
void range(F *a, int N, F const min, F const max);

__global__ void sigm_deriv_kernelf(float *out_err, float *act, float *in_err, int n);
__global__ void tanh_deriv_kernelf(float *out_err, float *act, float *in_err, int n);

__global__ void sigm_deriv_kerneld(double *out_err, double *act, double *in_err, int n);
__global__ void tanh_deriv_kerneld(double *out_err, double *act, double *in_err, int n);

template <typename F>
void sigm_deriv(F *out_err, F *act, F *in_err, int n);

template <typename F>
void tanh_deriv(F *out_err, F *act, F *in_err, int n);


#endif
