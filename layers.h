#ifndef __LAYERS_H__
#define __LAYERS_H__

#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>

#include "tensor.h"
#include "util.h"


struct ConvolutionLayer {
	ConvolutionLayer(int in_map, int out_map, int kw, int kh);
	~ConvolutionLayer();

	void init_normal(float mean, float std);

	void forward(Tensor &input, Tensor &output);

	void backward_weights(Tensor &input, Tensor &output_err);
	void backward_input(Tensor &output_err, Tensor &input_grad);


	cudnnConvolutionDescriptor_t conv;
	FilterBank filter_bank, filter_bank_grad;
	Tensor bias, bias_grad;
};

struct TanhLayer {
	void forward(Tensor &in, Tensor &out);
	void backward(Tensor &in, Tensor &out, Tensor &out_err, Tensor &in_err);
};

struct SoftmaxLayer {
	void forward(Tensor &in, Tensor &out);
	void backward(Tensor &in, Tensor &out_err, Tensor &in_err);
};

#endif
