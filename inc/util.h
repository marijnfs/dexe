#ifndef __UTIL_H__
#define __UTIL_H__

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <time.h>
#include <stdint.h>
#include <algorithm>
#include <stdint.h>

#include "handler.h"

struct StringException : public std::exception {
	StringException(std::string msg_): msg(msg_){}
	char const* what() const throw() {return msg.c_str();}
	~StringException() throw() {}
	std::string msg;
};

struct Timer {
	Timer() {start();}
	void start() {t = clock();}
	double since() {return double(clock() - t) / double(CLOCKS_PER_SEC);}

	clock_t t;
};

inline void handle_error(cublasStatus_t status) {
  switch (status)
    {
    case CUBLAS_STATUS_SUCCESS:
      return;

    case CUBLAS_STATUS_NOT_INITIALIZED:
      throw StringException("CUBLAS_STATUS_NOT_INITIALIZED");

    case CUBLAS_STATUS_ALLOC_FAILED:
      throw StringException("CUBLAS_STATUS_ALLOC_FAILED");

    case CUBLAS_STATUS_INVALID_VALUE:
      throw StringException("CUBLAS_STATUS_INVALID_VALUE");

    case CUBLAS_STATUS_ARCH_MISMATCH:
      throw StringException("CUBLAS_STATUS_ARCH_MISMATCH");

    case CUBLAS_STATUS_MAPPING_ERROR:
      throw StringException("CUBLAS_STATUS_MAPPING_ERROR");

    case CUBLAS_STATUS_EXECUTION_FAILED:
      throw StringException("CUBLAS_STATUS_EXECUTION_FAILED");

    case CUBLAS_STATUS_INTERNAL_ERROR:
      throw StringException("CUBLAS_STATUS_INTERNAL_ERROR");
	case CUBLAS_STATUS_NOT_SUPPORTED:
		throw StringException("CUBLAS_STATUS_NOT_SUPPORTED");
	default:
		throw StringException("SOME CUBLAS ERROR");

    }

  throw StringException("<unknown>");
}

inline void handle_error(curandStatus_t status) {
  switch(status) {
  case CURAND_STATUS_SUCCESS:
    return;
  case CURAND_STATUS_VERSION_MISMATCH:
    throw StringException("Header file and linked library version do not match");

  case CURAND_STATUS_NOT_INITIALIZED:
    throw StringException("Generator not initialized");
  case CURAND_STATUS_ALLOCATION_FAILED:
    throw StringException("Memory allocation failed");
  case CURAND_STATUS_TYPE_ERROR:
    throw StringException("Generator is wrong type");
  case CURAND_STATUS_OUT_OF_RANGE:
    throw StringException("Argument out of range");
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    throw StringException("Length requested is not a multple of dimension");
    // In CUDA >= 4.1 only
#if CUDART_VERSION >= 4010
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    throw StringException("GPU does not have double precision required by MRG32k3a");
#endif
  case CURAND_STATUS_LAUNCH_FAILURE:
    throw StringException("Kernel launch failure");
  case CURAND_STATUS_PREEXISTING_FAILURE:
    throw StringException("Preexisting failure on library entry");
  case CURAND_STATUS_INITIALIZATION_FAILED:
    throw StringException("Initialization of CUDA failed");
  case CURAND_STATUS_ARCH_MISMATCH:
    throw StringException("Architecture mismatch, GPU does not support requested feature");
  case CURAND_STATUS_INTERNAL_ERROR:
    throw StringException("Internal library error");
  default:
	  throw StringException("SOME CURAND ERROR");
  }

  throw StringException("Unknown error");
}

inline void handle_error(cudaError_t err) {
	if (err != cudaSuccess) {
		std::cerr << cudaGetErrorString(err) << std::endl;
		throw StringException(cudaGetErrorString(err));
	}
}


inline void handle_error(cudnnStatus_t status) {
	switch(status) {
	    case CUDNN_STATUS_SUCCESS:
			break;
		case CUDNN_STATUS_NOT_INITIALIZED:
			throw StringException("CUDNN_STATUS_NOT_INITIALIZED");
		case CUDNN_STATUS_ALLOC_FAILED:
			throw StringException("CUDNN_STATUS_ALLOC_FAILED");
		case CUDNN_STATUS_ARCH_MISMATCH:
			throw StringException("CUDNN_STATUS_ARCH_MISMATCH");
		case CUDNN_STATUS_MAPPING_ERROR:
			throw StringException("CUDNN_STATUS_MAPPING_ERROR");
		case CUDNN_STATUS_EXECUTION_FAILED:
			throw StringException("CUDNN_STATUS_EXECUTION_FAILED");
		case CUDNN_STATUS_INTERNAL_ERROR:
			throw StringException("CUDNN_STATUS_INTERNAL_ERROR");
		case CUDNN_STATUS_NOT_SUPPORTED:
			throw StringException("CUDNN_STATUS_NOT_SUPPORTED");
		case CUDNN_STATUS_LICENSE_ERROR:
			throw StringException("CUDNN_STATUS_LICENSE_ERROR");
     	default:
     		std::cerr << "err: " << status << std::endl;
	     	throw StringException("SOME CUDNN ERROR");
	}
}

template <typename F>
inline void add_cuda(F const *from, F *to, int n, F const alpha);

template <>
inline void add_cuda(float const *from, float *to, int n, float const alpha) {
  handle_error(cublasSaxpy(Handler::cublas(), n, &alpha, from, 1, to, 1));
}

template <>
inline void add_cuda<double>(double const *from, double *to, int n, double const alpha) {
  handle_error(cublasDaxpy(Handler::cublas(), n, &alpha, from, 1, to, 1));
}

template <typename F>
inline void scale_cuda(F *data, int n, F const alpha);

template <>
inline void scale_cuda(float *data, int n, float const alpha) {
	handle_error( cublasSscal(Handler::cublas(), n, &alpha, data, 1) );
}

template <>
inline void scale_cuda(double *data, int n, double const alpha) {
	handle_error( cublasDscal(Handler::cublas(), n, &alpha, data, 1) );
}

/* template <typename T> */
/* inline std::ostream &operator<<(std::ostream &out, std::vector<T> &in) { */
/* 	out << "["; */
/* 	typename std::vector<T>::const_iterator it = in.begin(), end = in.end(); */
/* 	for (; it != end; ++it) */
/* 		out << " " << *it; */
/* 	return out << "]"; */
/* } */

template <typename T>
inline std::ostream &operator<<(std::ostream &out, std::vector<T> in) {
  out << "[";
  typename std::vector<T>::const_iterator it = in.begin(), end = in.end();
  for (; it != end; ++it)
	  if (it == in.begin())
		  out << *it;
	  else
		  out << " " << *it;
  return out << "]";
}

template <typename T>
inline bool operator==(std::vector<T> &v1, std::vector<T> &v2) {
  if (v1.size() != v2.size())
    return false;
  for (size_t i(0); i < v1.size(); ++i)
    if (v1[i] != v2[i]) return false;
  return true;
}

template <typename T>
inline T &last(std::vector<T> &v) {
	return v[v.size() - 1];
}

template <typename T>
inline T &first(std::vector<T> &v) {
	return v[0];
}

template <typename T>
inline void del_vec(std::vector<T*> &v) {
	for (size_t i(0); i < v.size(); ++i)
		delete v[i];
}

template <typename T>
inline void fill(std::vector<T> &v, T val) {
	fill(v.begin(), v.end(), val);
}

template <typename T>
inline void random_shuffle(std::vector<T> &v) {
	random_shuffle(v.begin(), v.end());
}

template <typename T>
inline T abs(T a, bool bla = true) {
	return a > 0.0 ? a : -a;
}

template <typename T>
inline T norm(std::vector<T> &v) {
	T sum(0);
	typename std::vector<T>::const_iterator it(v.begin()), end(v.end());
	for (; it != end; ++it)
		sum += *it * *it;
	return sqrt(sum);
}


template <typename T>
inline T l1_norm(std::vector<T> &v) {
	T sum(0);
	typename std::vector<T>::const_iterator it(v.begin()), end(v.end());
	for (; it != end; ++it)
		sum += abs<T>(*it);
	return sum;
}


template <typename T>
inline void normalize(std::vector<T> *v) {
	float mean(0);
	for (size_t i(0); i < v->size(); ++i) mean += (*v)[i];
	mean /= v->size();
	for (size_t i(0); i < v->size(); ++i) (*v)[i] -= mean;
	float std(0);
	for (size_t i(0); i < v->size(); ++i) std += (*v)[i] * (*v)[i];
	std = sqrt(std / (v->size() - 1));
	for (size_t i(0); i < v->size(); ++i) (*v)[i] /= std;
}

template <typename T>
inline void normalize_1(std::vector<T> *v) {
	T sum = std::accumulate(v->begin(), v->end(), 0);
	for (size_t i(0); i < v->size(); ++i) (*v)[i] /= sum;
}


inline float rand_float() {
	return static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
}

struct Indices {
    Indices(int n) : indices(n) {
	    for (size_t i(0); i < n; ++i) indices[i] = i;
    }

	void shuffle() { random_shuffle(indices); }
	int operator[](int n) { return indices[n]; }

	std::vector<int> indices;
};

template <typename T>
inline void byte_write(std::ofstream &out, T &t) {
	out.write(reinterpret_cast<char*>(&t), sizeof(T));
}

template <typename T>
inline void byte_write_vec(std::ofstream &out, std::vector<T> &v) {
	uint64_t s(v.size());
	byte_write(out, s);
	for (size_t i(0); i < v.size(); ++i)
		byte_write(out, v[i]);
}

template <typename T>
inline T byte_read(std::ifstream &in) {
	T t;
	in.read(reinterpret_cast<char*>(&t), sizeof(T));
	return t;
}

template <typename T>
inline std::vector<T> byte_read_vec(std::ifstream &in) {
	size_t s = byte_read<uint64_t>(in);
	std::vector<T> v;
	for (size_t i(0); i < s; ++i)
		v.push_back(byte_read<T>(in));
	return v;
}

template <typename T>
void init_uniform(T *data, int n, T std);

__global__ void normal_kernel(int seed, float *data, int n, float mean, float std);
__global__ void normal_kerneld(int seed, double *data, int n, double mean, double std);

template <typename T>
void init_normal(T *data, int n, T mean, T std);

__global__ void rand_zero_kernel(int seed, float *data, int n, float p);
void rand_zero(float *data, int n, float p);


#endif
