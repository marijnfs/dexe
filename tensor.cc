#include <cassert>

#include "tensor.h"
#include "util.h"
#include "handler.h"

using namespace std;



Tensor::Tensor(int n_, int c_, int w_, int h_): 
  n(n_), w(w_), h(h_), c(c_), allocated(true)
{
	handle_error( cudnnCreateTensorDescriptor(&td));
	handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
	size_t even_size(((size() + 1) / 2) * 2); //we want multiple of two for curand
	handle_error( cudaMalloc( (void**)&data, sizeof(float) * even_size));
	if (ZERO_ON_INIT)
	  zero();
}

Tensor::Tensor(int n_, int c_, int w_, int h_, float *data_): 
  n(n_), w(w_), h(h_), c(c_), allocated(false), data(data_)
{
	handle_error( cudnnCreateTensorDescriptor(&td));
	handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
}

Tensor::Tensor(TensorShape s): 
  n(s.n), w(s.w), h(s.h), c(s.c), allocated(true)
{
	handle_error( cudnnCreateTensorDescriptor(&td));
	handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
	size_t even_size(((size() + 1) / 2) * 2); //we want multiple of two for curand
	handle_error( cudaMalloc( (void**)&data, sizeof(float) * even_size));
	if (ZERO_ON_INIT)
	  zero();
}

Tensor::Tensor(TensorShape s, float *data_): 
  n(s.n), w(s.w), h(s.h), c(s.c), allocated(false), data(data_)
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


vector<float> Tensor::to_vector() {
	vector<float> vec(n * c * h * w);
	handle_error( cudaMemcpy(&vec[0], data, vec.size() * sizeof(float), cudaMemcpyDeviceToHost));
	return vec;
}

void Tensor::from_vector(vector<float> &in) {
	assert(size() == in.size());
 	handle_error( cudaMemcpy(data, &in[0], in.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void Tensor::from_ptr(float const *in) {
	handle_error( cudaMemcpy(data, in, size() * sizeof(float), cudaMemcpyHostToDevice));	
}

void Tensor::init_normal(float mean, float std) {
	size_t even_size(((size() + 1) / 2) * 2);
  handle_error( curandGenerateNormal ( Handler::curand(), data, even_size, mean, std) );
}

int Tensor::size() const {
	return n * c * w * h;
}

TensorShape Tensor::shape() const {
	return TensorShape{n, c, w, h};
}

TensorSet::TensorSet(int n_, int c_, int w_, int h_) : 
	n(n_), c(c_), w(w_), h(h_), x(n, c, w, h), grad(n, c, w, h)
{
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
	handle_error( curandGenerateNormal ( Handler::curand(), weights, n_weights(), mean, std) );
}

vector<float> FilterBank::to_vector() {
	vector<float> vec(n_weights());
	handle_error( cudaMemcpy(&vec[0], weights, n_weights() * sizeof(float), cudaMemcpyDeviceToHost) );
	return vec;
}

void FilterBank::from_vector(vector<float> &in) {
	assert(n_weights() == in.size());
 	handle_error( cudaMemcpy(weights, &in[0], in.size() * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor &operator-=(Tensor &in, Tensor const &other) {
	assert(in.size() == other.size());
	add_cuda(other.data, in.data, in.size(), -1);
	return in;
}
