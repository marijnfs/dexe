#include "tensor.h"
#include "layers.h"
#include "handler.h"

using namespace std;

vector<float> Tensor::to_vector() {
	vector<float> vec(n * c * h * w);
	handle_error( cudaMemcpy(&vec[0], data, vec.size() * sizeof(float), cudaMemcpyDeviceToHost));
	return vec;
}

void Tensor::from_vector(vector<float> &in) {
  handle_error( cudaMemcpy(data, &in[0], in.size() * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor::Tensor(int n_, int w_, int h_, int c_): 
  n(n_), w(w_), h(h_), c(c_), allocated(true)
{
	handle_error( cudnnCreateTensorDescriptor(&td));
	handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
	handle_error( cudaMalloc( (void**)&data, sizeof(float) * n * c * h * w));
	if (ZERO_ON_INIT)
	  zero();
}

Tensor::Tensor(int n_, int w_, int h_, int c_, float *data_): 
  n(n_), w(w_), h(h_), c(c_), allocated(false), data(data_)
{
	handle_error( cudnnCreateTensorDescriptor(&td));
	handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
}


Tensor::~Tensor() {
	handle_error( cudnnDestroyTensorDescriptor(td));
	if (allocated)
	  cudaFree(data);
}

void Tensor::zero() {
  handle_error( cudaMemset(data, 0, sizeof(float) * n * w * h * c));
}


void Tensor::init_normal(float mean, float std) {
  handle_error( curandGenerateNormal ( Handler::curand(), data, n * w * h * c, mean, std ));
}


FilterBank::FilterBank(int in_map_, int out_map_, int kw_, int kh_): 
  in_map(in_map_), out_map(out_map_), kw(kw_), kh(kh_)
{
	handle_error( cudnnCreateFilterDescriptor(&fd));
	handle_error( cudnnSetFilter4dDescriptor(fd, CUDNN_DATA_FLOAT, out_map, in_map, kh, kw));
	handle_error(cudaMalloc( (void**)&weights, sizeof(float) * out_map * in_map * kw * kh));
}

FilterBank::~FilterBank() {
	cudnnDestroyFilterDescriptor(fd);
	cudaFree(weights);
}

void FilterBank::init_normal(float mean, float std) {
	handle_error( curandGenerateNormal ( Handler::curand(), weights, n_weights(), mean, std ));
}