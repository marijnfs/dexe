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
	ConvolutionLayer(int in_map, int out_map, int kw, int kh, Tensor &input);
	~ConvolutionLayer();

	void init(Tensor &input);
	void init_normal(float mean, float std);

	void forward(Tensor &input, Tensor &output);

	int in_map, out_map;
	int kw, kh;
	cudnnFilterDescriptor_t filter;
	cudnnConvolutionDescriptor_t conv;
	float *data;
};

struct TanhLayer {
	void forward(Tensor &in, Tensor &out);
};

struct SoftmaxLayer {
	void forward(Tensor &in, Tensor &out);
};

struct BiasLayer {
	void forward(Tensor &bias, Tensor &out);
};




#endif
