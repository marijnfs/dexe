#include "tensor.h"
#include "layers.h"

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




