#include "layers.h"
#include "handler.h"
#include <cublas_v2.h>


void Tensor::init_normal(float mean, float std) {
  handle_error( curandGenerateNormal ( Handler::curand(), data, n * w * h * c, mean, std ));
}

ConvolutionLayer::ConvolutionLayer(int in_map_, int out_map_, int kw_, int kh_):
	in_map(in_map_), out_map(out_map_), kw(kw_), kh(kh_)
{
	handle_error( cudnnCreateFilterDescriptor(&filter));
	handle_error( cudnnSetFilter4dDescriptor(filter, CUDNN_DATA_FLOAT, out_map, in_map, kh, kw));
	handle_error(cudaMalloc( (void**)&data, sizeof(float) * out_map * in_map * kw * kh));

	int pad_h(0), pad_w(0), stride_w(1), stride_h(1), upscalex(1), upscaley(1);
	handle_error( cudnnCreateConvolutionDescriptor(&conv));
	handle_error( cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride_h, stride_w, upscalex, upscaley, CUDNN_CONVOLUTION));
}

void ConvolutionLayer::init_normal(float mean, float std) {
  handle_error( curandGenerateNormal ( Handler::curand(), data, in_map * out_map * kh * kw, mean, std ));
}

void ConvolutionLayer::forward(Tensor &input, Tensor &output) {
  float alpha(1.0), beta(0.0);
  handle_error( cudnnConvolutionForward(Handler::cudnn(), &alpha, input.td, input.data, filter, data, conv, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 0, 0, (void*)&beta, output.td, (void*)output.data));
}

void ConvolutionLayer::forward(Tensor &input, Tensor &output, Tensor &bias) {
	forward(input, output);

	float alpha(1), beta(1);
	handle_error( cudnnAddTensor(Handler::cudnn(), CUDNN_ADD_SAME_C, &alpha, bias.td, bias.data, &beta, output.td, output.data));
}

// void ConvolutionLayer::backward(Tensor &diff_output, Tensor &diff_input) {
// 	float alpha(1.0), beta(0.0);
// 	handle_error( cudnnConvolutionBackward(Handler::cudnn(), &alpha, input.td, input.data, filter, data, conv, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 0, 0, (void*)&beta, output.td, (void*)output.data));	
// }

ConvolutionLayer::~ConvolutionLayer() {
	cudnnDestroyConvolutionDescriptor(conv);
	cudnnDestroyFilterDescriptor(filter);
	cudaFree(data);
}

void TanhLayer::forward(Tensor &in, Tensor &out) {
  float alpha(0), beta(0);
  handle_error( cudnnActivationForward(Handler::cudnn(), CUDNN_ACTIVATION_TANH, &alpha, in.td, in.data, &beta, out.td, out.data));
}


void SoftmaxLayer::forward(Tensor &in, Tensor &out) {
	float alpha(1), beta(0);
	handle_error( cudnnSoftmaxForward(Handler::cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, in.td, in.data, &beta, out.td, out.data));
}
