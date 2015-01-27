#include "layers.h"
#include "handler.h"
#include <cublas_v2.h>


ConvolutionLayer::ConvolutionLayer(int in_map, int out_map, int kw, int kh, bool keep):
	filter_bank(in_map, out_map, kw, kh),
	filter_bank_grad(in_map, out_map, kw, kh),
	bias(1, 1, 1, in_map * out_map),
	bias_grad(1, 1, 1, in_map * out_map)
{
	int pad_h(0), pad_w(0), stride_w(1), stride_h(1), upscalex(1), upscaley(1);
	if (keep) {
		pad_w = kw / 2;
		pad_h = kh / 2;
	}
	//todo: calculate padding
	handle_error( cudnnCreateConvolutionDescriptor(&conv));
	handle_error( cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride_h, stride_w, upscalex, upscaley, CUDNN_CONVOLUTION));
}

void ConvolutionLayer::init_normal(float mean, float std) {
	filter_bank.init_normal(mean, std);
	bias.init_normal(mean, std);
}

void ConvolutionLayer::forward(Tensor &input, Tensor &output) {
	float alpha(1.0), beta(0.0);
	handle_error( cudnnConvolutionForward(Handler::cudnn(), &alpha, input.td, input.data, filter_bank.fd, filter_bank.weights, conv, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 0, 0, (void*)&beta, output.td, (void*)output.data));

	float alpha_bias(1), beta_bias(1);
	handle_error( cudnnAddTensor(Handler::cudnn(), CUDNN_ADD_SAME_C, &alpha_bias, bias.td, bias.data, &beta_bias, output.td, output.data));
}

void ConvolutionLayer::backward_weights(Tensor &input, Tensor &output_err) {
	float alpha_bias(1.0), beta_bias(0.0);

	handle_error( cudnnConvolutionBackwardBias(Handler::cudnn(), &alpha_bias, output_err.td, output_err.data, &beta_bias, bias_grad.td, bias_grad.data) );

	float alpha(1.0), beta(1.0);
	handle_error( cudnnConvolutionBackwardFilter(Handler::cudnn(), &alpha, input.td, input.data, output_err.td, output_err.data, conv, &beta, filter_bank_grad.fd, filter_bank_grad.weights) );
}

void ConvolutionLayer::backward_input(Tensor &output_err, Tensor &input_err) {
	float alpha(1.0), beta(0.0);
	handle_error( cudnnConvolutionBackwardData(Handler::cudnn(), &alpha, filter_bank.fd, filter_bank.weights, output_err.td, output_err.data, conv, &beta, input_err.td, input_err.data) );
}


// void ConvolutionLayer::backward(Tensor &diff_output, Tensor &diff_input) {
// 	float alpha(1.0), beta(0.0);
// 	handle_error( cudnnConvolutionBackward(Handler::cudnn(), &alpha, input.td, input.data, filter_bank, data, conv, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 0, 0, (void*)&beta, output.td, (void*)output.data));	
// }

ConvolutionLayer::~ConvolutionLayer() {
	cudnnDestroyConvolutionDescriptor(conv);
}

SquashLayer::SquashLayer(Tensor &t, int n) : ConvolutionLayer(t.n, n, t.w, t.h, false) {

}

void TanhLayer::forward(Tensor &in, Tensor &out) {
  float alpha(1), beta(0);
  handle_error( cudnnActivationForward(Handler::cudnn(), CUDNN_ACTIVATION_TANH, &alpha, in.td, in.data, &beta, out.td, out.data));
}

void TanhLayer::backward(Tensor &in, Tensor &out, Tensor &out_err, Tensor &in_err) {
  float alpha(1), beta(0);
  handle_error( cudnnActivationBackward(Handler::cudnn(), CUDNN_ACTIVATION_TANH, &alpha, out.td, out.data, out_err.td, out_err.data, in.td, in.data, &beta, in_err.td, in_err.data));
}

void SoftmaxLayer::forward(Tensor &in, Tensor &out) {
	float alpha(1), beta(0);
	handle_error( cudnnSoftmaxForward(Handler::cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, in.td, in.data, &beta, out.td, out.data));
}

void SoftmaxLayer::backward(Tensor &in, Tensor &out_err, Tensor &in_err) {
	float alpha(1), beta(0);
	handle_error( cudnnSoftmaxBackward(Handler::cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, in.td, in.data, out_err.td, out_err.data, &beta, in_err.td, in_err.data));

}