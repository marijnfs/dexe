#include "operations.h"
#include "handler.h"
#include <cublas_v2.h>
#include <cassert>

using namespace std;

ConvolutionOperation::ConvolutionOperation(int in_map, int out_map, int kw, int kh, bool keep):
	filter_bank(in_map, out_map, kw, kh),
	filter_bank_grad(in_map, out_map, kw, kh),
	bias(1, out_map, 1, 1),
	bias_grad(1, out_map, 1, 1)
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

void ConvolutionOperation::update(float lr) {
	// cout << filter_bank_grad.to_vector() << endl;

	add_cuda<float>(filter_bank_grad.ptr(), filter_bank.ptr(), filter_bank.n_weights(), lr);
	add_cuda<float>(bias_grad.ptr(), bias.ptr(), bias.size(), lr * .1);
}

void ConvolutionOperation::l2(float l) {
	add_cuda<float>(filter_bank.ptr(), filter_bank_grad.ptr(), filter_bank.n_weights(), -l);
}

void ConvolutionOperation::init_normal(float mean, float std) {
	filter_bank.init_normal(mean, std);
	//bias.init_normal(mean, std);
}

vector<float> ConvolutionOperation::to_vector() {
	vector<float> filter_values = filter_bank.to_vector();
	vector<float> bias_values = bias.to_vector();
	copy(bias_values.begin(), bias_values.end(), back_inserter(filter_values));
	return filter_values;
}

void ConvolutionOperation::from_vector(vector<float> &v) {
	assert(v.size() == filter_bank.n_weights() + bias.size());
	vector<float> filter_bank_weights(v.begin(), v.begin() + filter_bank.n_weights());
	filter_bank.from_vector(filter_bank_weights);
	
	vector<float> bias_weights(v.begin() + filter_bank.n_weights(), v.begin() + filter_bank.n_weights() + bias.size());
	bias.from_vector(bias_weights);
}


vector<float> ConvolutionOperation::grad_to_vector() {
	vector<float> grad = filter_bank_grad.to_vector();
	vector<float> bias_grad_vec = bias_grad.to_vector();
	copy(bias_grad_vec.begin(), bias_grad_vec.end(), back_inserter(grad));
	return grad;
}

void ConvolutionOperation::forward(Tensor<float> &input, Tensor<float> &output) {
	float alpha(1.0), beta(0.0);

	float alpha_bias(1), beta_bias(1);
	handle_error( cudnnConvolutionForward(Handler::cudnn(), &alpha, input.td, input.data, filter_bank.fd, filter_bank.weights, conv, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 0, 0, &beta, output.td, output.data));

	handle_error( cudnnAddTensor(Handler::cudnn(), CUDNN_ADD_SAME_C, &alpha_bias, bias.td, bias.data, &beta_bias, output.td, output.data));
}

void ConvolutionOperation::backward_weights(Tensor<float> &input, Tensor<float> &output_grad) {
	float alpha_bias(1.0), beta_bias(0.0);
	handle_error( cudnnConvolutionBackwardBias(Handler::cudnn(), &alpha_bias, output_grad.td, output_grad.data, &beta_bias, bias_grad.td, bias_grad.data) );

	float alpha(1.0), beta(0.0);
	handle_error( cudnnConvolutionBackwardFilter(Handler::cudnn(), &alpha, input.td, input.data, output_grad.td, output_grad.data, conv, &beta, filter_bank_grad.fd, filter_bank_grad.weights) );
}

void ConvolutionOperation::backward(Tensor<float> &in, Tensor<float> &out, Tensor<float> &output_grad, Tensor<float> &input_grad) {
	float alpha(1.0), beta(0.0);
	handle_error( cudnnConvolutionBackwardData(Handler::cudnn(), &alpha, filter_bank.fd, filter_bank.weights, output_grad.td, output_grad.data, conv, &beta, input_grad.td, input_grad.data) );
}

TensorShape ConvolutionOperation::output_shape(TensorShape in) {
	return TensorShape{in.n, filter_bank.out_map, in.w, in.h};
}

ConvolutionOperation::~ConvolutionOperation() {
	cudnnDestroyConvolutionDescriptor(conv);
}

SquashOperation::SquashOperation(TensorShape s, int c_) : c(c_), ConvolutionOperation(s.c, c_, s.w, s.h, false) {

}

TensorShape SquashOperation::output_shape(TensorShape in) {
	return TensorShape{in.n, c, 1, 1};
}


PoolingOperation::PoolingOperation(int kw_, int kh_) : kw(kw_), kh(kh_) {
	handle_error( cudnnCreatePoolingDescriptor(&pool) );

	cudnnSetPooling2dDescriptor(pool, CUDNN_POOLING_MAX, kw, kh, 0, 0, kw, kh);
}

void PoolingOperation::forward(Tensor<float> &in, Tensor<float> &out) {
	float alpha(1.0), beta(0.0);
	handle_error( cudnnPoolingForward(Handler::cudnn(), pool, &alpha, in.td, in.data, &beta, out.td, out.data) );
}

void PoolingOperation::backward(Tensor<float> &in, Tensor<float> &out, Tensor<float> &out_grad, Tensor<float> &in_grad) {
	float alpha(1.0), beta(0.0);
	handle_error( cudnnPoolingBackward(Handler::cudnn(), pool, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data) );
}

TensorShape PoolingOperation::output_shape(TensorShape in) {
	cout << in.c << endl;
	return TensorShape{in.n, in.c, in.w / kw, in.h / kh};
}

void TanhOperation::forward(Tensor<float> &in, Tensor<float> &out) {
  float alpha(1), beta(0);
  handle_error( cudnnActivationForward(Handler::cudnn(), CUDNN_ACTIVATION_TANH, &alpha, in.td, in.data, &beta, out.td, out.data));
}

void TanhOperation::backward(Tensor<float> &in, Tensor<float> &out, Tensor<float> &out_grad, Tensor<float> &in_grad) {
  float alpha(1), beta(0);
  handle_error( cudnnActivationBackward(Handler::cudnn(), CUDNN_ACTIVATION_TANH, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data));
}

TensorShape TanhOperation::output_shape(TensorShape in) {
	return in;
}

void ReluOperation::forward(Tensor<float> &in, Tensor<float> &out) {
  float alpha(1), beta(0);
  handle_error( cudnnActivationForward(Handler::cudnn(), CUDNN_ACTIVATION_RELU, &alpha, in.td, in.data, &beta, out.td, out.data));
}

void ReluOperation::backward(Tensor<float> &in, Tensor<float> &out, Tensor<float> &out_grad, Tensor<float> &in_grad) {
  float alpha(1), beta(0);
  //handle_error( cudnnActivationBackward(Handler::cudnn(), CUDNN_ACTIVATION_RELU, &alpha, in.td, in.data, out_grad.td, out_grad.data, out.td, out.data, &beta, in_grad.td, in_grad.data));
  handle_error( cudnnActivationBackward(Handler::cudnn(), CUDNN_ACTIVATION_RELU, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data));
}

TensorShape ReluOperation::output_shape(TensorShape in) {
	return in;
}


SoftmaxOperation::SoftmaxOperation(bool matched_) : matched(matched_) {
}

void SoftmaxOperation::forward(Tensor<float> &in, Tensor<float> &out) {
	float alpha(1), beta(0);
	handle_error( cudnnSoftmaxForward(Handler::cudnn(), CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, in.td, in.data, &beta, out.td, out.data));
}

void SoftmaxOperation::backward(Tensor<float> &in, Tensor<float> &out, Tensor<float> &out_grad, Tensor<float> &in_grad) {
	float alpha(1), beta(0); 
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

TensorShape SoftmaxOperation::output_shape(TensorShape in) {
	return in;
}

