#ifndef __UTIL_H__
#define __UTIL_H__

#include <iostream>
#include <vector>
#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <time.h>
#include <stdint.h>
#include <algorithm>

#include "handler.h"

using namespace std;

struct StringException : public std::exception {
	StringException(std::string msg_): msg(msg_){}
	char const* what() const throw() {return msg.c_str();}
	~StringException() throw() {}
	std::string msg;
};

struct Timer {
	Timer() {t = clock();}
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
	}
}

inline void add_cuda(float const *from, float *to, int n, float const alpha) {
  handle_error(cublasSaxpy(Handler::cublas(), n, &alpha, from, 1, to, 1));
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
inline void random_shuffle(std::vector<T> &v) {
	random_shuffle(v.begin(), v.end());
}

#endif
