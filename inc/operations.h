#ifndef __OPERATIONS_H__
#define __OPERATIONS_H__

#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>

#include "tensor.h"
#include "util.h"

template <typename F>
struct Operation {
	virtual void forward(Tensor<F> &in, Tensor<F> &out){}

	virtual void backward_weights(Tensor<F> &in, Tensor<F> &out_grad){}
	virtual void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad){}
	virtual TensorShape output_shape(TensorShape input) { return TensorShape{0, 0, 0, 0}; }

	virtual void forward_dry_run(Tensor<F> &in, Tensor<F> &out){}
};

template <typename F>
struct Operation2 {
	virtual void forward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out){}

	virtual void backward_weights(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out_grad){}
	virtual void backward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, Tensor<F> &in2_grad){}
	virtual TensorShape output_shape(TensorShape input) { return TensorShape{0, 0, 0, 0}; }

	virtual void forward_dry_run(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out){}
};

template <typename F>
struct Parametrised {
	virtual void init_normal(F mean, F std) = 0;
	virtual void init_uniform(F var) = 0;
	virtual void update(F lr) {}
	virtual void l2(F l) {}
	virtual std::vector<F> to_vector() { return std::vector<F>(); }
	virtual void from_vector(std::vector<F> &v) { }
	virtual int size() { return 0; }
	virtual std::vector<F> grad_to_vector() { return std::vector<F>(); }
};

template <typename F>
struct ConvolutionOperation : public Operation<F>, public Parametrised<F> {
	ConvolutionOperation(int in_map, int out_map, int kw, int kh, bool keep = true, size_t workspace_limit = 0);
	~ConvolutionOperation();

	void init_normal(F mean, F std);
	void init_uniform(F var);

	void forward(Tensor<F> &in, Tensor<F> &out);

	void backward_weights(Tensor<F> &in, Tensor<F> &out_grad);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad);
	void update(F lr);
	void l2(F l);
	
	void forward_dry_run(Tensor<F> &in, Tensor<F> &out); // allocates workspace

	std::vector<F> to_vector();
	void from_vector(std::vector<F> &v);
	std::vector<F> grad_to_vector();
	int size();

	TensorShape output_shape(TensorShape input);

	cudnnConvolutionDescriptor_t conv;
	FilterBank<F> filter_bank, filter_bank_grad;
	Tensor<F> bias, bias_grad;

	cudnnConvolutionFwdAlgo_t algo;
	char *workspace;
	size_t workspace_size;
	bool keep;
};

template <typename F>
struct SquashOperation : ConvolutionOperation<F> {
	SquashOperation(TensorShape s, int c);
	TensorShape output_shape(TensorShape input);

	int c;
};

template <typename F>
struct GateOperation : public Operation2<F> {
	GateOperation();
	TensorShape output_shape(TensorShape input);

	virtual void forward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out);
	virtual void backward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, Tensor<F> &in2_grad);
};

template <typename F>
struct PoolingOperation : public Operation<F> {
	PoolingOperation(int kw, int kh);
	void forward(Tensor<F> &in, Tensor<F> &out);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad);

	TensorShape output_shape(TensorShape input);

	int kw, kh;
	cudnnPoolingDescriptor_t pool;
};

template <typename F>
struct TanhOperation : public Operation<F> {
	void forward(Tensor<F> &in, Tensor<F> &out);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad);

	TensorShape output_shape(TensorShape input);
};

template <typename F>
struct SigmoidOperation : public Operation<F> {
	void forward(Tensor<F> &in, Tensor<F> &out);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad);

	TensorShape output_shape(TensorShape input);
};

template <typename F>
struct STanhOperation : public Operation<F> {
	STanhOperation(TensorShape s);
	void forward(Tensor<F> &in, Tensor<F> &out);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad);

	TensorShape output_shape(TensorShape input);
	TensorSet<F> tmp;
};

template <typename F>
struct ReluOperation : public Operation<F> {
	void forward(Tensor<F> &in, Tensor<F> &out);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad);

	TensorShape output_shape(TensorShape input);
};

template <typename F>
struct SoftmaxOperation : public Operation<F> {
	SoftmaxOperation(bool matched = true);
	void forward(Tensor<F> &in, Tensor<F> &out);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad);

	TensorShape output_shape(TensorShape input);
	bool matched;
};

#endif
