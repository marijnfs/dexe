#pragma once

#include <cudnn.h>
#include <vector>
#include <iostream>
#include "util.h"

const bool ZERO_ON_INIT(true);
const cudnnTensorFormat_t DEFAULT_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
// const cudnnTensorFormat_t DEFAULT_TENSOR_FORMAT = CUDNN_TENSOR_NHWC;

namespace dexe {

struct TensorShape {
  std::vector<int> dimensions;


  TensorShape(){}
  TensorShape(int n, int c, int h, int w);
  TensorShape(int n, int c, int d, int h, int w);
  explicit TensorShape(std::vector<int> dimensions);

  bool operator==(TensorShape const &other) const;
  bool operator!=(TensorShape const &other) const;
  int &operator[](int index);

  int offset(int n, int c, int y, int x);
  int n_elements();
  int n_dimensions();
  int n_pixels();

  void set_c(int c);

    template<class Archive>
	void serialize(Archive & archive)
  	{
    	archive( dimensions ); // serialize things by passing them to the archive
 	}


  int n();
  int c();
  int d();
  int h(); 
  int w();
};

template <typename F>
struct Tensor {
	Tensor();
	Tensor(TensorShape shape, cudnnTensorFormat_t format = DEFAULT_TENSOR_FORMAT);
	Tensor(TensorShape shape, F *data, cudnnTensorFormat_t format = DEFAULT_TENSOR_FORMAT);
	Tensor(std::vector<F> data); //creates a single dim tensor with c = len(data)


	~Tensor();

	void allocate();
    void set_descriptor_typed();
    void set_descriptor();
	void reshape(TensorShape shape);

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
	void from_tensor(Tensor<F> &in, F alpha = 1.0);
	void fill(F val);
   

	F asum();
	F sum();
	F mean();
	F norm();
	F norm2();
	void threshold(F value);

	Tensor<F> &operator*=(F val);
	Tensor<F> &operator/=(F val);
	Tensor<F> &operator-=(F val);
	Tensor<F> &operator+=(Tensor<F> &t);
	Tensor<F> &operator*=(Tensor<F> &t);
	Tensor<F> &operator-=(Tensor<F> &t);

  	int size();

  	F *ptr() { return data; }
    F *ptr(int n_, int c_ = 0, int y_ = 0, int x_ = 0) {return data + shape.offset(n_, c_, y_, x_); }
   
   	void add(Tensor<F> &other, F alpha);
   	void scale(F alpha);

   	template<class Archive>
	void save(Archive & archive)
  	{
  		auto cpu_data = to_vector();
    	archive( shape, format, cpu_data ); // serialize things by passing them to the archive
 	}

   	template<class Archive>
	void load(Archive & archive)
  	{
  		std::vector<F> data;
    	archive( shape, data, format ); // serialize things by passing them to the archive
		allocate();
		set_descriptor();

    	//init
    	from_vector(data);
 	}

    TensorShape shape;
	bool owning = false;
	cudnnTensorDescriptor_t td = nullptr;
	F *data = nullptr;

	cudnnTensorFormat_t format = DEFAULT_TENSOR_FORMAT;
};

template <typename F>
inline Tensor<F> &operator*=(Tensor<F> &in, F const other) {
  scale_cuda<F>(in.data, in.size(), other);
	return in;
}

template <typename F>
struct TensorSet {
	TensorSet(TensorShape shape);
	TensorSet();

	void alloc_x(TensorShape shape);
	void alloc_grad(TensorShape shape);
	TensorShape shape();


	std::unique_ptr<Tensor<F>> x, grad;
};

template <typename F>
struct FilterBank {
	FilterBank();
	FilterBank(std::vector<int> dimensions);
	~FilterBank();

	std::vector<int> dimensions;

	cudnnFilterDescriptor_t fd = 0;

	F *weights = nullptr;

	void init();

	void reshape(std::vector<int> dimensions);

	int n_weights() { return calculate_product(dimensions); }	
	void init_normal(F mean, F std);
	void init_uniform(F var);

	int in_c();
	int out_c();
	int kd();
	int kh();
	int kw();

	std::vector<F> to_vector();
	void from_vector(std::vector<F> &in);
	void fill(F val);
	void zero();

	template<class Archive>
	void save(Archive & archive)
  	{
  		auto cpu_data = to_vector();
    	archive( dimensions, cpu_data ); // serialize things by passing them to the archive
 	}

   	template<class Archive>
	void load(Archive & archive)
  	{
  		std::vector<F> cpu_data;
  		archive(dimensions, cpu_data);
  		init();
  		from_vector(cpu_data);
 	}

	F *ptr() { return weights; }
};

}

inline std::ostream &operator<<(std::ostream &o, dexe::TensorShape s) {
	return o << "T:" << s.dimensions;
}

template <typename F>
inline std::ostream &operator<<(std::ostream &o, dexe::FilterBank<F> &f) {
  return o << "F:" << f.dimensions;
}


