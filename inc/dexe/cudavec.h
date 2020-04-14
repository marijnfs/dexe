#pragma once

#include "util.h"

#include <cuda.h>
#include <vector>

#include <iostream>

extern uint64_t memory_counter;

namespace dexe {


template <typename F> struct DEXE_API CudaVec {
    F *data = 0;
    int N = 0;
    bool own = true;

    CudaVec();
    CudaVec(int n_);
    CudaVec(F *data, int n_);

    ~CudaVec();

    CudaVec &operator=(CudaVec &other);

    void allocate(int n);

    bool allocated();

    void free();

    CudaVec(CudaVec &other);

    void zero(int offset = 0);

    void init_normal(F mean, F std);
    void add_normal(F mean, F std);

    void share(CudaVec<F> &other);

    std::vector<F> to_vector();

    void to_ptr(F *target);

    void from_ptr(F const *source);

    void from_vector(std::vector<F> const &vec);

    F sum();

    CudaVec<F> &sqrt();
    CudaVec<F> &abs();
    CudaVec<F> &pow(F e);
    CudaVec<F> &exp();
    CudaVec<F> &clip(F limit);
    CudaVec<F> &add(int idx, F val);

    CudaVec<F> &operator-=(CudaVec<F> &other);
    CudaVec<F> &operator+=(CudaVec<F> &other);
    CudaVec<F> &operator*=(CudaVec<F> &other);
    CudaVec<F> &operator/=(CudaVec<F> &other);
    CudaVec<F> &operator/=(F val);

    CudaVec &operator*=(F v);
    CudaVec &operator+=(F v);

    // Remove copy operator to be safe
    CudaVec<F>(const CudaVec<F> &) = delete;
};

__global__ void sqrt_kernel(float *v, int n);
__global__ void pow_kernel(float *v, int n, float e);
__global__ void times_kernel(float *v, float *other, int n);
__global__ void add_scalar(float *v, float other, int n);
__global__ void times_scalar(float *v, float other, int n);
__global__ void divide_kernel(float *v, float *other, int n);
__global__ void abs_kernel(float *v, int n);
__global__ void exp_kernel(float *v, int n);
__global__ void clip_kernel(float *v, int n, float limit);

__global__ void sqrt_kerneld(double *v, int n);
__global__ void pow_kerneld(double *v, int n, double exp);
__global__ void times_kerneld(double *v, double *other, int n);
__global__ void add_scalard(double *v, double other, int n);
__global__ void times_scalard(double *v, double other, int n);
__global__ void divide_kerneld(double *v, double *other, int n);
__global__ void abs_kerneld(double *v, int n);
__global__ void exp_kerneld(double *v, int n);
__global__ void clip_kerneld(double *v, int n, double limit);

} // namespace dexe
