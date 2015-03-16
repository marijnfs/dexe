#include "operations.h"
#include "handler.h"
#include "gate.h"
#include <cublas_v2.h>
#include <cassert>

using namespace std;

template <typename F>
ConvolutionOperation<F>::ConvolutionOperation(int in_map, int out_map, int kw, int kh, bool keep_, size_t workspace_limit_):
	filter_bank(in_map, out_map, kw, kh),
	filter_bank_grad(in_map, out_map, kw, kh),
	bias(1, out_map, 1, 1),
	bias_grad(1, out_map, 1, 1),
	algo(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM), //default algorithm
	workspace(0),
	workspace_size(workspace_limit_),
	keep(keep_)
{
	int pad_h(0), pad_w(0), stride_w(1), stride_h(1), upscalex(1), upscaley(1);
	if (keep) {
		pad_w = kw / 2;
		pad_h = kh / 2;
	}
	cout << "weight buffer: " << filter_bank.n_weights() << endl;
	cout << "bias buffer: " << bias.size() << endl;
	//todo: calculate padding
	handle_error( cudnnCreateConvolutionDescriptor(&conv));

	handle_error( cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride_h, stride_w, upscalex, upscaley, CUDNN_CROSS_CORRELATION));
	//handle_error( cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride_h, stride_w, upscalex, upscaley, CUDNN_CONVOLUTION));
}

template <typename F>
void ConvolutionOperation<F>::update(F lr) {
	// cout << filter_bank_grad.to_vector() << endl;

	add_cuda<F>(filter_bank_grad.ptr(), filter_bank.ptr(), filter_bank.n_weights(), lr);
	add_cuda<F>(bias_grad.ptr(), bias.ptr(), bias.size(), lr * .1);
}

template <typename F>
void ConvolutionOperation<F>::l2(F l) {
	add_cuda<F>(filter_bank.ptr(), filter_bank_grad.ptr(), filter_bank.n_weights(), -l);
}

template <typename F>
void ConvolutionOperation<F>::init_normal(F mean, F std) {
	filter_bank.init_normal(mean, std);
	//bias.init_normal(mean, std);
}

template <typename F>
void ConvolutionOperation<F>::init_uniform(F var) {
	filter_bank.init_uniform(var);
	bias.init_uniform(var);
}

template <typename F>
vector<F> ConvolutionOperation<F>::to_vector() {
	vector<F> filter_values = filter_bank.to_vector();
	vector<F> bias_values = bias.to_vector();
	copy(bias_values.begin(), bias_values.end(), back_inserter(filter_values));
	return filter_values;
}

template <typename F>
void ConvolutionOperation<F>::from_vector(vector<F> &v) {
	assert(v.size() == filter_bank.n_weights() + bias.size());
	vector<F> filter_bank_weights(v.begin(), v.begin() + filter_bank.n_weights());
	filter_bank.from_vector(filter_bank_weights);
	
	vector<F> bias_weights(v.begin() + filter_bank.n_weights(), v.begin() + filter_bank.n_weights() + bias.size());
	bias.from_vector(bias_weights);
}

template <typename F>
int ConvolutionOperation<F>::size() {
	return filter_bank.n_weights() + bias.size();
}

template <typename F>
vector<F> ConvolutionOperation<F>::grad_to_vector() {
	vector<F> grad = filter_bank_grad.to_vector();
	vector<F> bias_grad_vec = bias_grad.to_vector();
	copy(bias_grad_vec.begin(), bias_grad_vec.end(), back_inserter(grad));
	return grad;
}

template <typename F>
void ConvolutionOperation<F>::forward(Tensor<F> &input, Tensor<F> &output) {
	F alpha(1.0), beta(0.0);

	F alpha_bias(1), beta_bias(1);
	
	handle_error( cudnnConvolutionForward(Handler::cudnn(), &alpha, input.td, input.data, filter_bank.fd, filter_bank.weights, conv, algo, workspace, workspace_size, &beta, output.td, output.data));

	handle_error( cudnnAddTensor(Handler::cudnn(), CUDNN_ADD_SAME_C, &alpha_bias, bias.td, bias.data, &beta_bias, output.td, output.data));
}

template <typename F>
void ConvolutionOperation<F>::backward_weights(Tensor<F> &input, Tensor<F> &output_grad) {
	F alpha_bias(1.0), beta_bias(0.0);
	handle_error( cudnnConvolutionBackwardBias(Handler::cudnn(), &alpha_bias, output_grad.td, output_grad.data, &beta_bias, bias_grad.td, bias_grad.data) );

	F alpha(1.0), beta(0.0);
	handle_error( cudnnConvolutionBackwardFilter(Handler::cudnn(), &alpha, input.td, input.data, output_grad.td, output_grad.data, conv, &beta, filter_bank_grad.fd, filter_bank_grad.weights) );
}

template <typename F>
void ConvolutionOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &output_grad, Tensor<F> &input_grad) {
	F alpha(1.0), beta(0.0);
	handle_error( cudnnConvolutionBackwardData(Handler::cudnn(), &alpha, filter_bank.fd, filter_bank.weights, output_grad.td, output_grad.data, conv, &beta, input_grad.td, input_grad.data) );
}

template <typename F>
TensorShape ConvolutionOperation<F>::output_shape(TensorShape in) {
	int x_even((filter_bank.kw + 1) % 2), y_even((filter_bank.kh + 1) % 2);
	return TensorShape{in.n, filter_bank.out_map, in.w + x_even, in.h + y_even};
}

template <typename F>
void ConvolutionOperation<F>::forward_dry_run(Tensor<F> &in, Tensor<F> &out) { // allocates workspace
	//handle_error( cudnnGetConvolutionForwardAlgorithm(Handler::cudnn(), in.td, filter_bank.fd, conv, out.td, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, workspace_size, &algo) );
	algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
	handle_error( cudnnGetConvolutionForwardWorkspaceSize(Handler::cudnn(), in.td, filter_bank.fd, conv, out.td, algo, &workspace_size) );
	cout << "workspace size: " << workspace_size << endl;
	if (workspace_size)
		handle_error( cudaMalloc( (void**)&workspace, workspace_size) );
}

template <typename F>
ConvolutionOperation<F>::~ConvolutionOperation() {
	cudnnDestroyConvolutionDescriptor(conv);

    if (workspace)
		cudaFree(workspace);
}

template <typename F>
SquashOperation<F>::SquashOperation(TensorShape s, int c_) : c(c_), ConvolutionOperation<F>(s.c, c_, s.w, s.h, false) {

}

template <typename F>
TensorShape SquashOperation<F>::output_shape(TensorShape in) {
	return TensorShape{in.n, c, 1, 1};
}

template <typename F>
PoolingOperation<F>::PoolingOperation(int kw_, int kh_) : kw(kw_), kh(kh_) {
	handle_error( cudnnCreatePoolingDescriptor(&pool) );

	cudnnSetPooling2dDescriptor(pool, CUDNN_POOLING_MAX, kw, kh, 0, 0, kw, kh);
}

template <typename F>
void PoolingOperation<F>::forward(Tensor<F> &in, Tensor<F> &out) {
	F alpha(1.0), beta(0.0);
	handle_error( cudnnPoolingForward(Handler::cudnn(), pool, &alpha, in.td, in.data, &beta, out.td, out.data) );
}

template <typename F>
void PoolingOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad) {
	F alpha(1.0), beta(0.0);
	handle_error( cudnnPoolingBackward(Handler::cudnn(), pool, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data) );
}

template <typename F>
TensorShape PoolingOperation<F>::output_shape(TensorShape in) {
	cout << in.c << endl;
	return TensorShape{in.n, in.c, in.w / kw, in.h / kh};
}

template <typename F>
void TanhOperation<F>::forward(Tensor<F> &in, Tensor<F> &out) {
  F alpha(1), beta(0);
  handle_error( cudnnActivationForward(Handler::cudnn(), CUDNN_ACTIVATION_TANH, &alpha, in.td, in.data, &beta, out.td, out.data));
}

template <typename F>
void TanhOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad) {
  F alpha(1), beta(0);
  handle_error( cudnnActivationBackward(Handler::cudnn(), CUDNN_ACTIVATION_TANH, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data));
}

template <typename F>
TensorShape TanhOperation<F>::output_shape(TensorShape in) {
	return in;
}

template <typename F>
void GateOperation<F>::forward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out) {
	gate(in, in2, out);
}

template <typename F>
void GateOperation<F>::backward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, Tensor<F> &in2_grad) {
	gate(out_grad, in2, in_grad);
	gate(out_grad, in, in2_grad);
}

template <typename F>
TensorShape GateOperation<F>::output_shape(TensorShape in) {
	return in;
}


template <typename F>
void SigmoidOperation<F>::forward(Tensor<F> &in, Tensor<F> &out) {
  F alpha(1), beta(0);
  handle_error( cudnnActivationForward(Handler::cudnn(), CUDNN_ACTIVATION_SIGMOID, &alpha, in.td, in.data, &beta, out.td, out.data));
}

template <typename F>
void SigmoidOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad) {
  F alpha(1), beta(0);
  handle_error( cudnnActivationBackward(Handler::cudnn(), CUDNN_ACTIVATION_SIGMOID, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data));
}

template <typename F>
TensorShape SigmoidOperation<F>::output_shape(TensorShape in) {
	return in;
}


template <typename F>
STanhOperation<F>::STanhOperation(TensorShape s) : tmp(s) {
}

template <typename F>
void STanhOperation<F>::forward(Tensor<F> &in, Tensor<F> &out) {
  F alpha(1.7159), beta(0);
  tmp.x.from_tensor(in);
  scale_cuda(tmp.x.data, tmp.x.size(), 2./3.);
  handle_error( cudnnActivationForward(Handler::cudnn(), CUDNN_ACTIVATION_TANH, &alpha, tmp.x.td, tmp.x.data, &beta, out.td, out.data));
}

template <typename F>
void STanhOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad) {
  F alpha(1.0), beta(0);
  tmp.x.from_tensor(out);  
  scale_cuda(tmp.x.data, tmp.x.size(), 1.0 / 1.7159);
  handle_error( cudnnActivationBackward(Handler::cudnn(), CUDNN_ACTIVATION_TANH, &alpha, tmp.x.td, tmp.x.data, out_grad.td, out_grad.data, tmp.x.td, tmp.x.data, &beta, in_grad.td, in_grad.data));
  scale_cuda(in_grad.data, in_grad.size(), 3./2.);
}

template <typename F>
TensorShape STanhOperation<F>::output_shape(TensorShape in) {
	return in;
}

template <typename F>
void ReluOperation<F>::forward(Tensor<F> &in, Tensor<F> &out) {
  F alpha(1), beta(0);
  handle_error( cudnnActivationForward(Handler::cudnn(), CUDNN_ACTIVATION_RELU, &alpha, in.td, in.data, &beta, out.td, out.data));
}

template <typename F>
void ReluOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad) {
  F alpha(1), beta(0);
  //handle_error( cudnnActivationBackward(Handler::cudnn(), CUDNN_ACTIVATION_RELU, &alpha, in.td, in.data, out_grad.td, out_grad.data, out.td, out.data, &beta, in_grad.td, in_grad.data));
  handle_error( cudnnActivationBackward(Handler::cudnn(), CUDNN_ACTIVATION_RELU, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data));
}

template <typename F>
TensorShape ReluOperation<F>::output_shape(TensorShape in) {
	return in;
}

template <typename F>
SoftmaxOperation<F>::SoftmaxOperation(bool matched_) : matched(matched_) {
}

template <typename F>
void SoftmaxOperation<F>::forward(Tensor<F> &in, Tensor<F> &out) {
	F alpha(1), beta(0);
	handle_error( cudnnSoftmaxForward(Handler::cudnn(), CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, in.td, in.data, &beta, out.td, out.data));
}

template <typename F>
void SoftmaxOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad) {
	F alpha(1), beta(0); 
	//cout << out_grad.to_vector() << endl;
	//cout << in_grad.to_vector() << endl;
	//cout << out.to_vector() << endl;
	//cout << in.to_vector() << endl;

	if (matched) {//loss function matched 
		in_grad.from_tensor(out_grad);
	}
	else		
		handle_error( cudnnSoftmaxBackward(Handler::cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, out.td, out.data, out_grad.td, out_grad.data, &beta, in_grad.td, in_grad.data));
}

template <typename F>
TensorShape SoftmaxOperation<F>::output_shape(TensorShape in) {
	return in;
}

template struct ConvolutionOperation<float>;
template struct SquashOperation<float>;
template struct PoolingOperation<float>;
template struct TanhOperation<float>;
template struct SigmoidOperation<float>;
template struct ReluOperation<float>;
template struct SoftmaxOperation<float>;
template struct GateOperation<float>;

template struct ConvolutionOperation<double>;
template struct SquashOperation<double>;
template struct PoolingOperation<double>;
template struct TanhOperation<double>;
template struct SigmoidOperation<double>;
template struct ReluOperation<double>;
template struct SoftmaxOperation<double>;
template struct GateOperation<double>;


