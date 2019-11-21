#pragma once

#include "tensor.h"

namespace dexe {

__global__ void split_kernelf(size_t const N, size_t const C, size_t const X, size_t const Y, float const *input, float *out);
__global__ void split_kerneld(size_t const N, size_t const C, size_t const X, size_t const Y, double const *input, double *out);

void split(Tensor<float> &a, Tensor<float> &out);
void split(Tensor<double> &a, Tensor<double> &out);


__global__ void merge_kernelf(size_t const N, size_t const C, size_t const X, size_t const Y, float const *input, float *out);
__global__ void merge_kerneld(size_t const N, size_t const C, size_t const X, size_t const Y, double const *input, double *out);


void merge(Tensor<float> &a, Tensor<float> &out);
void merge(Tensor<double> &a, Tensor<double> &out);

__global__ void gate_kernelf(size_t N, float const *a, float const *b, float *out);
__global__ void gate_kerneld(size_t N, double const *a, double const *b, double *out);

template <typename F>
void gate(Tensor<F> &a, Tensor<F> &b, Tensor<F> &out);

__global__ void gateinv_kernelf(size_t N, float const *a, float const *b, float *out);
__global__ void gateinv_kerneld(size_t N, double const *a, double const *b, double *out);

template <typename F>
void gateinv(Tensor<F> &a, Tensor<F> &b, Tensor<F> &out);

__global__ void range_kernelf(float *a, size_t N, float const min, float const max);
__global__ void range_kerneld(double *a, size_t N, double const min, double const max);

template <typename F>
void range(F *a, size_t N, F const min, F const max);

__global__ void sigm_forward_kernelf(float *in, float *out, size_t n, float beta, float scale);
__global__ void sigm_forward_kerneld(double *in, double *out, size_t n, double beta, double scale);

template <typename F>
void sigm_forward(F *in, F *out, size_t n, F beta, F scale);


__global__ void sigm_deriv_kernelf(float *out_err, float *act, float *in_err, size_t n, float beta, float scale);
__global__ void sigm_deriv_kerneld(double *out_err, double *act, double *in_err, size_t n, double beta, double scale);

template <typename F>
void sigm_deriv(F *out_err, F *act, F *in_err, size_t n, F beta, F scale);



__global__ void tanh_forward_kernelf(float *in, float *out, size_t n, float beta, float scale); 
__global__ void tanh_forward_kerneld(double *in, double *out, size_t n, double beta, double scale);

template <typename F>
void tanh_forward(F *in, F *out, size_t n, F beta, F scale);


__global__ void tanh_deriv_kernelf(float *out_err, float *act, float *in_err, size_t n, float beta, float scale);
__global__ void tanh_deriv_kerneld(double *out_err, double *act, double *in_err, size_t n, double beta, double scale);

template <typename F>
void tanh_deriv(F *out_err, F *act, F *in_err, size_t n, F beta, F scale);


template <typename F>
__global__ void support_kernel(F *prediction, F *target, F *loss, size_t N, F support);

template <typename F>
void support_loss(F *input, F *target, F *loss, size_t N, F support);

template <typename F>
__global__ void threshold_kernel(F *input, size_t N, F threshold);

template <typename F>
void threshold_cuda(F *input, size_t N, F threshold);

}
