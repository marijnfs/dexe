#ifndef __OPERATIONS_H__
#define __OPERATIONS_H__

#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>

#include "tensor.h"
#include "util.h"

struct Operation {
	virtual void forward(Tensor<float> &in, Tensor<float> &out){}

	virtual void backward_weights(Tensor<float> &in, Tensor<float> &out_grad){}
	virtual void backward(Tensor<float> &in, Tensor<float> &out, Tensor<float> &out_grad, Tensor<float> &in_grad){}
	virtual TensorShape output_shape(TensorShape input) { return TensorShape{0, 0, 0, 0}; }
};

struct Parametrised {
	virtual void init_normal(float mean, float std) = 0;
	virtual void update(float lr) {}
	virtual void l2(float l) {}
	virtual std::vector<float> to_vector() { return std::vector<float>(); }
	virtual void from_vector(std::vector<float> &v) { }
	virtual std::vector<float> grad_to_vector() { return std::vector<float>(); }
};

struct ConvolutionOperation : public Operation, public Parametrised {
	ConvolutionOperation(int in_map, int out_map, int kw, int kh, bool keep = true);
	~ConvolutionOperation();

	void init_normal(float mean, float std);

	void forward(Tensor<float> &in, Tensor<float> &out);

	void backward_weights(Tensor<float> &in, Tensor<float> &out_grad);
	void backward(Tensor<float> &in, Tensor<float> &out, Tensor<float> &out_grad, Tensor<float> &in_grad);
	void update(float lr);
	void l2(float l);

	std::vector<float> to_vector();
	void from_vector(std::vector<float> &v);
	std::vector<float> grad_to_vector();

	TensorShape output_shape(TensorShape input);

	cudnnConvolutionDescriptor_t conv;
	FilterBank<float> filter_bank, filter_bank_grad;
	Tensor<float> bias, bias_grad;
};

struct SquashOperation : ConvolutionOperation {
	SquashOperation(TensorShape s, int c);
	TensorShape output_shape(TensorShape input);

	int c;
};

struct PoolingOperation : public Operation {
	PoolingOperation(int kw, int kh);
	void forward(Tensor<float> &in, Tensor<float> &out);
	void backward(Tensor<float> &in, Tensor<float> &out, Tensor<float> &out_grad, Tensor<float> &in_grad);

	TensorShape output_shape(TensorShape input);

	int kw, kh;
	cudnnPoolingDescriptor_t pool;
};

struct TanhOperation : public Operation {
	void forward(Tensor<float> &in, Tensor<float> &out);
	void backward(Tensor<float> &in, Tensor<float> &out, Tensor<float> &out_grad, Tensor<float> &in_grad);

	TensorShape output_shape(TensorShape input);
};

struct ReluOperation : public Operation {
	void forward(Tensor<float> &in, Tensor<float> &out);
	void backward(Tensor<float> &in, Tensor<float> &out, Tensor<float> &out_grad, Tensor<float> &in_grad);

	TensorShape output_shape(TensorShape input);
};

struct SoftmaxOperation : public Operation {
	SoftmaxOperation(bool matched = true);
	void forward(Tensor<float> &in, Tensor<float> &out);
	void backward(Tensor<float> &in, Tensor<float> &out, Tensor<float> &out_grad, Tensor<float> &in_grad);

	TensorShape output_shape(TensorShape input);
	bool matched;
};

#endif
