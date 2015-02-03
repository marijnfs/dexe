#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <cudnn.h>
#include <vector>
#include <iostream>

const bool ZERO_ON_INIT(true);

struct TensorShape {
	int n, c, w, h;
};

struct Tensor {
	Tensor(int n, int c, int w, int h);
	Tensor(int n, int c, int w, int h, float *data);
	Tensor(TensorShape shape);
	Tensor(TensorShape shape, float *data);
	~Tensor();

	void init_normal(float mean, float std);
	void zero();
	
	std::vector<float> to_vector();
	void from_vector(std::vector<float> &in);
	void from_ptr(float const *in);
	void from_tensor(Tensor &in);
	void fill(float val);

  	int size() const;
	TensorShape shape() const;

	float *ptr() { return data; }

	int n, c, w, h;
	bool allocated;
	cudnnTensorDescriptor_t td;
	float *data;

};

Tensor &operator-=(Tensor &in, Tensor const &other);


struct TensorSet {
	Tensor x, grad;

	TensorSet(int n, int c, int w, int h);
	TensorSet(Tensor shape);
	TensorShape shape() const;

	int n, c, w, h;
};

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
	void fill(float val);
	void zero();

	float *ptr() { return weights; }

};

inline std::ostream &operator<<(std::ostream &o, TensorShape s) {	
	return o << "[" << s.n << "," << s.c << "," << s.w << "," << s.h << "]";
}


#endif
