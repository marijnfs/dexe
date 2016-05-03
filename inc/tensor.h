#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <cudnn.h>
#include <vector>
#include <iostream>
#include "util.h"

const bool ZERO_ON_INIT(true);

struct TensorShape {
	int n, c, w, h;
};

template <typename F>
struct Tensor {
	Tensor(int n = 0, int c = 0, int w = 0, int h = 0);
	Tensor(int n, int c, int w, int h, F *data);
	Tensor(TensorShape shape);
	Tensor(TensorShape shape, F *data);
	~Tensor();

	void init_normal(F mean, F std);
	void init_uniform(F var);
	void zero();

	std::vector<F> to_vector();
	void to_ptr(F *ptr);
	void from_vector(std::vector<F> &in);
	void from_ptr(F const *in);
	void from_tensor(Tensor &in);
	void fill(F val);
	void write_img(std::string filename);

	void reshape(TensorShape shape);
	void reshape(int n, int c, int w, int h);

	F sum();
	F norm();
	F norm2();

  	int size() const;
	TensorShape shape() const;

	F *ptr() { return data; }


	int n, c, w, h;
	bool allocated;
	cudnnTensorDescriptor_t td;
	F *data;
};

template <typename F>
Tensor<F> &operator-=(Tensor<F> &in, Tensor<F> const &other);

template <typename F>
inline Tensor<F> &operator*=(Tensor<F> &in, float const other) {
	scale_cuda<F>(in.data, in.n, other);
	return in;
}

template <typename F>
struct TensorSet {
	Tensor<F> x, grad;

	TensorSet(int n, int c, int w, int h);
	TensorSet(TensorShape shape);
	TensorShape shape() const;

	int n, c, w, h;
};

template <typename F>
struct FilterBank {
	FilterBank(int in_map_, int out_map_, int kw_, int kh_, int T = 1); //T is for rolledout filter bank
	~FilterBank();
	int in_map, out_map;
	int kw, kh;
	int N, T;
	cudnnFilterDescriptor_t fd;

	F *weights;

	int n_weights() { return N * T; }
	void init_normal(F mean, F std);
	void init_uniform(F var);

	std::vector<F> to_vector();
	void from_vector(std::vector<F> &in);
	void fill(F val);
	void zero();
	void draw_filterbank(std::string filename);

	F *ptr(int n = 0) { return weights + n * N; }

};

inline std::ostream &operator<<(std::ostream &o, TensorShape s) {
	return o << "[" << s.n << "," << s.c << "," << s.w << "," << s.h << "]";
}


#endif
