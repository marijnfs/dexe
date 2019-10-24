#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <cudnn.h>
#include <vector>
#include <iostream>
#include "util.h"
#include "normalise.h"

const bool ZERO_ON_INIT(true);

struct TensorShape {
  int n = 0, c = 0, w = 0, h = 0;
 
  TensorShape(){}
  TensorShape(int n, int c, int w, int h);
  bool operator==(TensorShape const &other) const;
  bool operator!=(TensorShape const &other) const;

  int offset(int n, int c, int y, int x);
  int size();
};

template <typename F>
struct Tensor {
	Tensor(int n = 0, int c = 0, int w = 0, int h = 0);
	Tensor(int n, int c, int w, int h, F *data);
	Tensor(TensorShape shape);
	Tensor(TensorShape shape, F *data);

	~Tensor();

	//Remove copy and assignment operator to be safe
	Tensor<F> & operator=(const Tensor<F>&) = delete;
    Tensor<F>(const Tensor<F>&) = delete;

	void init_normal(F mean, F std);
	void init_uniform(F var);
	void zero();

	std::vector<F> to_vector();
	void to_ptr(F *ptr);
	void from_vector(std::vector<F> &in);
	void from_ptr(F const *in);
	void from_tensor(Tensor &in);
	void fill(F val);
  	void write_img(std::string filename, int c = 0);
   
	void reshape(TensorShape shape);
	void reshape(int n, int c, int w, int h);
  
	F sum();
	F norm();
	F norm2();

  	int size();

  	F *ptr() { return data; }
    F *ptr(int n_, int c_ = 0, int y_ = 0, int x_ = 0) {return data + shape.offset(n_, c_, y_, x_); }
   

    TensorShape shape;
	bool owning = false;
	cudnnTensorDescriptor_t td;
	F *data;
};

template <typename F>
Tensor<F> &operator-=(Tensor<F> &in, Tensor<F> const &other);

template <typename F>
inline Tensor<F> &operator*=(Tensor<F> &in, float const other) {
  scale_cuda<F>(in.data, in.size(), other);
	return in;
}

template <typename F>
struct TensorSet {
	std::unique_ptr<Tensor<F>> x, grad;

	TensorSet(TensorShape shape);
	TensorSet(){}

	void alloc_x();
	void alloc_grad();

	TensorShape shape;
};

template <typename F>
struct FilterBank {
	FilterBank(int in_c_, int out_c_, int kw_, int kh_, int T = 1); //T is for rolledout filter bank
	~FilterBank();
	int in_c, out_c;
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
	return o << "[n: " << s.n << ", c:" << s.c << ", w:" << s.w << ", h:" << s.h << "]";
}

template <typename F>
inline std::ostream &operator<<(std::ostream &o, FilterBank<F> &f) {
  return o << "[in_c: " << f.in_c << ">out_c:" << f.out_c << " kw:" << f.kw << " kh:" << f.kh << " N:" << f.N << " T:" << f.T << "]";
}


#endif
