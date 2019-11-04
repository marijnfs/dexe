#ifndef __CUDAVEC_H__
#define __CUDAVEC_H__

#include <vector>
#include <cuda.h>
#include "util.h"

template <typename F>
struct CudaVec {
	F *data = 0;
	int N = 0;

	CudaVec() : data(0), N(0) { }
	CudaVec(int n_) : data(0), N(0) { resize(n_); }
	~CudaVec() {
	  if (N) {
	    cudaFree(data);
	  }
	}	  
	
	void resize(int n2) {
		if (N != n2) {
          if (N) {
            cudaFree(data);
          }
          handle_error( cudaMalloc( (void**)&data, sizeof(F) * n2));
          N = n2;
		}
		zero();
	}

	CudaVec(CudaVec &other) {
      resize(other.N);  
      copy_gpu_to_gpu(other.data, data, N);
	}

  CudaVec &operator=(CudaVec &other) {
    if (N != other.N) {
      resize(other.N);
    }
    copy_gpu_to_gpu(other.data, data, N);
    return *this;
  }
    
	void rand_zero(F p);

  void zero(int offset = 0) {
		handle_error( cudaMemset(data + offset, 0, sizeof(F) * (N - offset) ) );
	}

	void init_normal(F mean, F std) {
		::init_normal<F>(data, N, mean, std);
	}

  	void add_normal(F mean, F std) {
		::add_normal<F>(data, N, mean, std);
	}

	std::vector<F> to_vector() {
		std::vector<F> vec(N);
		handle_error( cudaMemcpy(&vec[0], data, N * sizeof(F), cudaMemcpyDeviceToHost));
		return vec;
	}

	void from_vector(std::vector<F> &vec) {
		if (vec.size() != N)
			resize(vec.size());
		handle_error( cudaMemcpy(data, &vec[0], N * sizeof(F), cudaMemcpyHostToDevice));
	}

  F sum() {
    F result(0);
    if (sizeof(F) == sizeof(float))
	    handle_error( cublasSasum(Handler::cublas(), N, data, 1, &result) );
    else
    	handle_error( cublasDasum(Handler::cublas(), N, data, 1, &result) );
    return result;
  }
  
  
  
	CudaVec &sqrt();
	CudaVec &abs();
	CudaVec &pow(F e);
	CudaVec &exp();
	CudaVec &clip(F limit);
	CudaVec &add(int idx, F val);

	CudaVec &operator-=(CudaVec &other);
	CudaVec &operator+=(CudaVec &other);
	CudaVec &operator*=(CudaVec &other);
  CudaVec &operator/=(CudaVec &other);
  CudaVec &operator/=(F val);


	CudaVec &operator*=(F v);
	CudaVec &operator+=(F v);



};

// __global__ void sqrt_kernel(float *v, int n);
// __global__ void abs_kernel(float *v, int n);
// __global__ void pow_kernel(float *v, int n);
// __global__ void exp_kernel(float *v, int n);
// __global__ void clip_kernel(float *v, int n, float limit);

// __global__ void times_kernel(float *v, float *other, int n);
// __global__ void divide_kernel(float *v, float *other, int n);
// __global__ void times_scalarf(float *v, float other, int n);
// __global__ void add_scalarf(float *v, float other, int n);



#endif
