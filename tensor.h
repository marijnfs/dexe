#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <cudnn.h>
#include <vector>

const bool ZERO_ON_INIT(true);

struct Tensor {
	Tensor(int n, int w, int h, int c);
	Tensor(int n, int w, int h, int c, float *data);
	~Tensor();
	void init_normal(float mean, float std);
	void zero();
	std::vector<float> to_vector();
	void from_vector(std::vector<float> &in);
  
	int n, w, h, c;
	bool allocated;
	cudnnTensorDescriptor_t td;
	float *data;

};

template <typename T=float>
struct Tensor2D {
	Tensor2D(int w, int h);
	Tensor2D(int w, int h, T *data);
	~Tensor2D();
	void zero();
	std::vector<T> to_vector();
	void from_vector(std::vector<T> &in);
	
	int w, h;
	bool allocated;
	T *data;	
};

#include "util.h"
#include "tensor2d.cc"

#endif
