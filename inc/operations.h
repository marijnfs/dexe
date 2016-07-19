#ifndef __OPERATIONS_H__
#define __OPERATIONS_H__

#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>
#include <string>

#include "tensor.h"
#include "util.h"
#include "cudaptr.h"

//int const CONV_MAX_MEM = 0;
int const CONV_MAX_MEM = 64 * 1024 * 1024;


template <typename F>
struct Operation {
	virtual void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0){}

	virtual void backward_weights(Tensor<F> &in, Tensor<F> &out_grad, F beta = 0.0){}
	virtual void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta = 0.0){}
	virtual TensorShape output_shape(TensorShape input) { return TensorShape{0, 0, 0, 0}; }

	virtual void forward_dry_run(Tensor<F> &in, Tensor<F> &out){}
	virtual void describe(std::ostream &out){}


	virtual void forward_timed(Tensor<F> &in, Tensor<F> &out, int t, F beta = 0.0){ forward(in, out, beta); }
	virtual void backward_weights_timed(Tensor<F> &in, Tensor<F> &out_grad, int t, F beta = 0.0){}
	virtual void backward_timed(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, int t, F beta = 0.0){}
};

template <typename F>
struct Operation2 {
	virtual void forward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out, F beta = 0.0){}

	virtual void backward_weights(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out_grad){}
	virtual void backward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, Tensor<F> &in2_grad, F beta = 0.0){}

	virtual TensorShape output_shape(TensorShape input) { return TensorShape{0, 0, 0, 0}; }

	virtual void forward_dry_run(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out){}
	virtual void describe(std::ostream &out){}

};

template <typename F>
struct Parametrised {
	virtual void init_normal(F mean, F std) = 0;
	virtual void init_uniform(F var) = 0;
	virtual void update(F lr) {}
	virtual void l2(F l) {}
	virtual void zero_grad() {}
	virtual void scale_grad(float val) {}
	virtual void register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &fast_params, std::vector<CudaPtr<F>> &grads, std::vector<CudaPtr<F> > &fast_grads) {}

	virtual std::vector<F> to_vector() { return std::vector<F>(); }
	virtual void from_vector(std::vector<F> &v) { }
	virtual int size() { return 0; }
	virtual std::vector<F> grad_to_vector() { return std::vector<F>(); }
};

template <typename F>
struct ConvolutionOperation : public Operation<F>, public Parametrised<F> {
	ConvolutionOperation(int in_map, int out_map, int kw, int kh, bool keep = true, size_t workspace_limit = CONV_MAX_MEM);// 64*1024*1024);
	ConvolutionOperation(std::string tmp, int in_map, int out_map, int kw, int kh, int z, bool keep = true, size_t workspace_limit = CONV_MAX_MEM);//32*1024*1024);

	~ConvolutionOperation();

	void init_normal(F mean, F std);
	void init_uniform(F var);

	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);

	void backward_weights(Tensor<F> &in, Tensor<F> &out_grad, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta = 0.0);
	void update(F lr);
	void l2(F l);
	void zero_grad();
	void scale_grad(F val);
	void register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &fast_params, std::vector<CudaPtr<F>> &grads, std::vector<CudaPtr<F> > &fast_grads);
	void share(ConvolutionOperation<F> &other);

	void forward_dry_run(Tensor<F> &in, Tensor<F> &out); // allocates workspace

	void forward_timed(Tensor<F> &in, Tensor<F> &out, int t, F beta = 0.0);
	void backward_weights_timed(Tensor<F> &in, Tensor<F> &out_grad, int t, F beta = 0.0);
	void backward_timed(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, int t, F beta = 0.0);


	std::vector<F> to_vector();
	void from_vector(std::vector<F> &v);
	std::vector<F> grad_to_vector();
	virtual int size();

	TensorShape output_shape(TensorShape input);
	void describe(std::ostream &out) { out << "conv " << kw << "x" << kh << " " << in_map << "->" << out_map; }

	int kw, kh, in_map, out_map;

	cudnnConvolutionDescriptor_t conv;
	FilterBank<F> filter_bank, filter_bank_grad;
	Tensor<F> bias, bias_grad;

	cudnnConvolutionFwdAlgo_t algo;
	cudnnConvolutionBwdDataAlgo_t algo_bwd;
	cudnnConvolutionBwdFilterAlgo_t algo_bwd_filter;

	char *workspace, *workspace_bwd, *workspace_bwd_filter;
	size_t workspace_size, workspace_size_bwd, workspace_size_bwd_filter;

	bool keep, rollout;
};

template <typename F>
struct ConvolutionShiftOperation : public ConvolutionOperation<F> {
	ConvolutionShiftOperation(int in_map, int out_map, int kw, int kh, int shift_x, int shift_y, bool keep = true, size_t workspace_limit = 0);
	~ConvolutionShiftOperation();

	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta = 0.0);

	void backward_weights(Tensor<F> &in, Tensor<F> &out_grad, F beta = 0.0);
	void forward_dry_run(Tensor<F> &in, Tensor<F> &out); // allocates workspace
	void zero_grad();

	int dx, dy;
	Tensor<F> slate, slate_grad;
};

template <typename F>
struct SquashOperation : ConvolutionOperation<F> {
	SquashOperation(TensorShape s, int c);
	TensorShape output_shape(TensorShape input);

	int c;
	void describe(std::ostream &out) { out << "squash"; }
};

template <typename F>
struct GateOperation : public Operation2<F> {
	//GateOperation() {}
	TensorShape output_shape(TensorShape input);

	virtual void forward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out, F beta = 0.0);
	virtual void backward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, Tensor<F> &in2_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "gate"; }

};

template <typename F>
struct PoolingOperation : public Operation<F> {
	PoolingOperation(int kw, int kh);
	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "pool " << kw << "x" << kh; }

	TensorShape output_shape(TensorShape input);

	int kw, kh;
	cudnnPoolingDescriptor_t pool;
};

template <typename F>
struct TanhOperation : public Operation<F> {
  TanhOperation(F scale = 1.0);
	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "tanh"; }

	TensorShape output_shape(TensorShape input);

	cudnnActivationDescriptor_t desc;

	F scale;
};

template <typename F>
struct SigmoidOperation : public Operation<F> {
  SigmoidOperation(F scale = 1.0);
	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "sigmoid"; }

	TensorShape output_shape(TensorShape input);
	cudnnActivationDescriptor_t desc;

	F scale;
};

template <typename F>
struct STanhOperation : public Operation<F> {
	STanhOperation(TensorShape s);
	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "stanh"; }

	TensorShape output_shape(TensorShape input);
	TensorSet<F> tmp;
};

template <typename F>
struct ReluOperation : public Operation<F> {
		ReluOperation();

	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "relu"; }

	TensorShape output_shape(TensorShape input);
	cudnnActivationDescriptor_t desc;

};


template <typename F>
struct SoftmaxOperation : public Operation<F> {
	SoftmaxOperation(bool matched = false);
	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "softmax"; }

	TensorShape output_shape(TensorShape input);
	bool matched;
};

#endif
