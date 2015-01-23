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

struct FilterBank {
	FilterBank(int in_map_, int out_map_, int kw_, int kh_);
	~FilterBank();
	int in_map, out_map;
	int kw, kh;
	cudnnFilterDescriptor_t fd;

	float *weights;

	int n_weights() { return in_map * out_map * kw * kh; }
};

#endif
