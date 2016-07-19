#ifndef __CUDAVEC_H__
#define __CUDAVEC_H__

#include <vector>
#include <cuda.h>
#include "util.h"

struct CudaVec {
	float *data;
	int n;

	CudaVec() : data(0), n(0) { }
	CudaVec(int n_) : data(0), n(0) { resize(n_); }
	~CudaVec() {
	  if (n) {
	    std::cout <<"deallocating " << n << std::endl;
	    cudaFree(data);
	  }
	}	  
	void resize(int n2) {
		if (n != n2) {
			std::cout << "freeing " << n << std::endl;
			cudaFree(data);
			std::cout <<"allocating " << n2 << std::endl;
			handle_error( cudaMalloc( (void**)&data, sizeof(float) * n2));
			n = n2;
		}
		zero();
	}

	CudaVec(CudaVec &other) {
	  //CudaVec &operator=(CudaVec &other) {
	  if (n != other.n) {
	    resize(other.n);
	  }
	  n = other.n;
	  copy_gpu_to_gpu(other.data, data, n);
	}

	void rand_zero(float p);

	void zero(int offset = 0) {
		handle_error( cudaMemset(data + offset, 0, sizeof(float) * (n - offset) ) );
	}

	void init_normal(float mean, float std) {
		::init_normal<float>(data, n, mean, std);
	}

  	void add_normal(float mean, float std) {
		::add_normal<float>(data, n, mean, std);
	}

	std::vector<float> to_vector() {
		std::vector<float> vec(n);
		handle_error( cudaMemcpy(&vec[0], data, n * sizeof(float), cudaMemcpyDeviceToHost));
		return vec;
	}

	void from_vector(std::vector<float> &vec) {
		if (vec.size() != n)
			resize(vec.size());
		handle_error( cudaMemcpy(data, &vec[0], n * sizeof(float), cudaMemcpyHostToDevice));
	}


	CudaVec &sqrt();
	CudaVec &abs();
	CudaVec &pow(float e);
	CudaVec &exp();
	CudaVec &clip(float limit);
	CudaVec &add(int idx, float val);

	CudaVec &operator-=(CudaVec &other);
	CudaVec &operator+=(CudaVec &other);
	CudaVec &operator*=(CudaVec &other);
	CudaVec &operator/=(CudaVec &other);


	CudaVec &operator*=(float v);
	CudaVec &operator+=(float v);



};

__global__ void sqrt_kernel(float *v, int n);
__global__ void abs_kernel(float *v, int n);
__global__ void pow_kernel(float *v, int n);
__global__ void exp_kernel(float *v, int n);
__global__ void clip_kernel(float *v, int n, float limit);

__global__ void times_kernel(float *v, float *other, int n);
__global__ void divide_kernel(float *v, float *other, int n);
__global__ void times_scalarf(float *v, float other, int n);
__global__ void add_scalarf(float *v, float other, int n);



#endif
