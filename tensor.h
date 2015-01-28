#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <cudnn.h>
#include <vector>

const bool ZERO_ON_INIT(true);

struct Tensor {
	Tensor(int n, int c, int w, int h);
	Tensor(int n, int c, int w, int h, float *data);
	~Tensor();

	void init_normal(float mean, float std);
	void zero();
	
	std::vector<float> to_vector();
	void from_vector(std::vector<float> &in);
	void from_ptr(float const *in);
  	int size() const;

	float *ptr() { return data; }

	int n, w, h, c;
	bool allocated;
	cudnnTensorDescriptor_t td;
	float *data;

};

Tensor &operator-=(Tensor &in, Tensor const &other);

struct FilterBank {
	FilterBank(int in_map_, int out_map_, int kw_, int kh_);
	~FilterBank();
	int in_map, out_map;
	int kw, kh;
	cudnnFilterDescriptor_t fd;

	float *weights;

	int n_weights() { return in_map * out_map * kw * kh; }
	void init_normal(float mean, float std);
	std::vector<float> to_vector();
	void from_vector(std::vector<float> &in);

	float *ptr() { return weights; }

};

#endif
