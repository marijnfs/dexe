#include <cassert>
#include <cstdlib>

#include "tensor.h"
#include "util.h"
#include "handler.h"
#include "img.h"

using namespace std;


template <>
Tensor<float>::Tensor(int n_, int c_, int w_, int h_):
  n(n_), w(w_), h(h_), c(c_), allocated(true)
{
	handle_error( cudnnCreateTensorDescriptor(&td));

	//size_t even_size(((size() + 1) / 2) * 2); //we want multiple of two for curand
	// size_t even_size(size());
	// handle_error( cudaMalloc( (void**)&data, sizeof(float) * even_size));
	if (n * w * h * c != 0) {
		handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
		handle_error( cudaMalloc( (void**)&data, sizeof(float) * size()));
		if (ZERO_ON_INIT)
		  zero();
	}
}

template <>
Tensor<float>::Tensor(int n_, int c_, int w_, int h_, float *data_):
  n(n_), w(w_), h(h_), c(c_), allocated(false), data(data_)
{
	handle_error( cudnnCreateTensorDescriptor(&td));
	if (n * w * h * c != 0) {
		handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
	}
}

template <>
Tensor<float>::Tensor(TensorShape s):
  n(s.n), w(s.w), h(s.h), c(s.c), allocated(true)
{
	handle_error( cudnnCreateTensorDescriptor(&td));

	////size_t even_size(((size() + 1) / 2) * 2); //we want multiple of two for curand
	//size_t even_size(size());

	if (n * w * h * c != 0) {
		handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
		handle_error( cudaMalloc( (void**)&data, sizeof(float) * size()));
		if (ZERO_ON_INIT)
		  zero();
	}
}

template <>
Tensor<float>::Tensor(TensorShape s, float *data_):
  n(s.n), w(s.w), h(s.h), c(s.c), allocated(false), data(data_)
{
	handle_error( cudnnCreateTensorDescriptor(&td));
	if (n * w * h * c != 0) {
		handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
	}
}

template <>
Tensor<double>::Tensor(int n_, int c_, int w_, int h_):
  n(n_), w(w_), h(h_), c(c_), allocated(true)
{
	handle_error( cudnnCreateTensorDescriptor(&td));

	if (n * w * h * c != 0) {
		handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
		//size_t even_size(((size() + 1) / 2) * 2); //we want multiple of two for curand
		// size_t even_size(size());
		handle_error( cudaMalloc( (void**)&data, sizeof(double) * size()));
		if (ZERO_ON_INIT)
		  zero();
	}
}

template <>
void Tensor<float>::reshape(int n_, int c_, int w_, int h_) {
	if (allocated)
		if (n * c * w * h != 0)
			cudaFree(data);

	n = n_;
	c = c_;
	w = w_;
	h = h_;

	// handle_error( cudnnDestroyTensorDescriptor(td));
	handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason

	if (allocated) {
		handle_error( cudaMalloc( (void**)&data, sizeof(float) * size()));
		if (ZERO_ON_INIT)
			zero();
	}
}




template <>
Tensor<double>::Tensor(int n_, int c_, int w_, int h_, double *data_):
  n(n_), w(w_), h(h_), c(c_), allocated(false), data(data_)
{
	handle_error( cudnnCreateTensorDescriptor(&td));
	handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
}

template <>
Tensor<double>::Tensor(TensorShape s):
  n(s.n), w(s.w), h(s.h), c(s.c), allocated(true)
{
	handle_error( cudnnCreateTensorDescriptor(&td));
	handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
	//size_t even_size(((size() + 1) / 2) * 2); //we want multiple of two for curand
	size_t even_size(size());
	handle_error( cudaMalloc( (void**)&data, sizeof(double) * even_size));
	if (ZERO_ON_INIT)
	  zero();
}

template <>
void Tensor<double>::reshape(int n_, int c_, int w_, int h_) {
	if (allocated)
		if (n * c * w * h != 0)
			cudaFree(data);

	n = n_;
	c = c_;
	w = w_;
	h = h_;

	// handle_error( cudnnDestroyTensorDescriptor(td));
	handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason

	if (allocated) {
		handle_error( cudaMalloc( (void**)&data, sizeof(float) * size()));
		if (ZERO_ON_INIT)
			zero();
	}
}

template <>
Tensor<double>::Tensor(TensorShape s, double *data_):
  n(s.n), w(s.w), h(s.h), c(s.c), allocated(false), data(data_)
{
	handle_error( cudnnCreateTensorDescriptor(&td));
	handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
}

template <typename F>
Tensor<F>::~Tensor() {
	handle_error( cudnnDestroyTensorDescriptor(td));
	if (allocated)
	  cudaFree(data);
}

template <typename F>
void Tensor<F>::reshape(TensorShape shape) {
	reshape(shape.n, shape.c, shape.w, shape.h);
}


template <typename F>
void Tensor<F>::zero() {
	handle_error( cudaMemset(data, 0, sizeof(F) * size()));
}


template <typename F>
vector<F> Tensor<F>::to_vector() {
	vector<F> vec(n * c * h * w);
	handle_error( cudaMemcpy(&vec[0], data, vec.size() * sizeof(F), cudaMemcpyDeviceToHost));
	return vec;
}

template <typename F>
void Tensor<F>::to_ptr(F *ptr) {
	handle_error( cudaMemcpy(ptr, data, size() * sizeof(F), cudaMemcpyDeviceToHost));
}

template <typename F>
void Tensor<F>::from_vector(vector<F> &in) {
	if (size() != in.size()) {
		throw StringException("sizes don't match");
	}
 	handle_error( cudaMemcpy(data, &in[0], in.size() * sizeof(F), cudaMemcpyHostToDevice));
}

template <typename F>
void Tensor<F>::from_tensor(Tensor &in) {
	if (size() != in.size()) {
 			throw StringException("sizes don't match");
	}
	handle_error( cudaMemcpy(data, in.data, in.size() * sizeof(F), cudaMemcpyDeviceToDevice));
}

template <typename F>
void Tensor<F>::from_ptr(F const *in) {
	handle_error( cudaMemcpy(data, in, size() * sizeof(F), cudaMemcpyHostToDevice));
}

template <>
void Tensor<float>::init_normal(float mean, float std) {
	//size_t even_size(((size() + 1) / 2) * 2);
	::init_normal(data, size(), mean, std);
	// size_t even_size(size());
	// handle_error( curandGenerateNormal(Handler::curand(), data, even_size, mean, std) );
}

template <>
void Tensor<double>::init_normal(double mean, double std) {
	//size_t even_size(((size() + 1) / 2) * 2);
	::init_normal(data, size(), mean, std);
	// size_t even_size(size());
	// handle_error( curandGenerateNormalDouble(Handler::curand(), data, even_size, mean, std) );
}

template <typename F>
void Tensor<F>::init_uniform(F var) {
  ::init_uniform(data, size(), var);
  /*
	vector<F> vec = to_vector();
	for (size_t i(0); i < vec.size(); ++i)
		vec[i] = -var + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)/(2.0 * var));
	from_vector(vec);
  */
}

template <typename F>
void Tensor<F>::fill(F val) {
	vector<F> vals(size());
	::fill<F>(vals, val);
	from_vector(vals);
}

template <typename F>
int Tensor<F>::size() const {
	return n * c * w * h;
}

template <>
float Tensor<float>::norm() {
	float result(0);
	handle_error( cublasSdot(Handler::cublas(), size(), data, 1, data, 1, &result) );
	return sqrt(result);
}

template <>
double Tensor<double>::norm() {
	double result(0);
	handle_error( cublasDdot(Handler::cublas(), size(), data, 1, data, 1, &result) );
	return sqrt(result);
}

template <>
float Tensor<float>::norm2() {
	float result(0);
	handle_error( cublasSdot(Handler::cublas(), size(), data, 1, data, 1, &result) );
	return result;
}

template <>
double Tensor<double>::norm2() {
	double result(0);
	handle_error( cublasDdot(Handler::cublas(), size(), data, 1, data, 1, &result) );
	return result;
}


template <>
float Tensor<float>::sum() {
	float result(0);
	handle_error( cublasSasum(Handler::cublas(), size(), data, 1, &result) );
	return result;
}

template <>
double Tensor<double>::sum() {
	double result(0);
	handle_error( cublasDasum(Handler::cublas(), size(), data, 1, &result) );
	return result;
}

template <typename F>
void Tensor<F>::write_img(string filename) {
	vector<F> v = to_vector();
	vector<float> vf(v.size());

	for (size_t i(0); i < v.size(); ++i)
		vf[i] = v[(i * h * w) % (h * w * c) + (i / c)];
	::write_img(filename, c, w, h, &vf[0]);
}

template <typename F>
TensorShape Tensor<F>::shape() const {
	return TensorShape{n, c, w, h};
}

template <typename F>
TensorSet<F>::TensorSet(int n_, int c_, int w_, int h_) :
	n(n_), c(c_), w(w_), h(h_), x(n_, c_, w_, h_), grad(n_, c_, w_, h_)
{
}

template <typename F>
TensorSet<F>::TensorSet(TensorShape s) : n(s.n), c(s.c), w(s.w), h(s.h), x(s.n, s.c, s.w, s.h), grad(s.n, s.c, s.w, s.h) {
	cout << "created set with shape: " << x.shape() << endl;
}

template <typename F>
TensorShape TensorSet<F>::shape() const {
	return x.shape();
}

template <>
FilterBank<float>::FilterBank(int in_map_, int out_map_, int kw_, int kh_, int T_):
  in_map(in_map_), out_map(out_map_), kw(kw_), kh(kh_),
  T(T_), N(in_map_ * out_map_ * kw_ * kh_)
{
	handle_error( cudnnCreateFilterDescriptor(&fd));
	handle_error( cudnnSetFilter4dDescriptor(fd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_map, in_map, kh, kw));
	handle_error( cudaMalloc( (void**)&weights, sizeof(float) * T * N));
	if (ZERO_ON_INIT)
	  zero();
}

template <>
FilterBank<double>::FilterBank(int in_map_, int out_map_, int kw_, int kh_, int T_):
  in_map(in_map_), out_map(out_map_), kw(kw_), kh(kh_),
  T(T_), N(in_map_ * out_map_ * kw_ * kh_)
{
	handle_error( cudnnCreateFilterDescriptor(&fd));
	handle_error( cudnnSetFilter4dDescriptor(fd, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, out_map, in_map, kh, kw));
	handle_error( cudaMalloc( (void**)&weights, sizeof(double) * T * N));
	if (ZERO_ON_INIT)
	  zero();
}

template <typename F>
FilterBank<F>::~FilterBank() {
	cudnnDestroyFilterDescriptor(fd);
	cudaFree(weights);
}

template <>
void FilterBank<float>::init_normal(float mean, float std) {
	::init_normal(weights, n_weights(), mean, std);
}

template <>
void FilterBank<double>::init_normal(double mean, double std) {
	::init_normal(weights, n_weights(), mean, std);
	// size_t even_size(((n_weights() + 1) / 2) * 2);
	// size_t even_size(n_weights());
	// handle_error( curandGenerateNormalDouble ( Handler::curand(), weights, even_size, mean, std) );
}

template <typename F>
void FilterBank<F>::init_uniform(F var) {
	::init_uniform(weights, n_weights(), var);
	// zero();
	// vector<F> vec = to_vector();
	// for (size_t i(0); i < vec.size(); ++i)
	// 	vec[i] = -var + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2 * var)));
	// from_vector(vec);
}

template <typename F>
void FilterBank<F>::zero() {
	// cout << "zero: " << in_map << " " << out_map << " " << kw << " " << kh << " " << N << " " << T  << " " << n_weights() << " " << weights << endl;
	handle_error( cudaMemset(weights, 0, sizeof(F) * n_weights()));
}

template <typename F>
vector<F> FilterBank<F>::to_vector() {
	vector<F> vec(n_weights());
	handle_error( cudaMemcpy(&vec[0], weights, n_weights() * sizeof(F), cudaMemcpyDeviceToHost) );
	return vec;
}

template <typename F>
void FilterBank<F>::from_vector(vector<F> &in) {
	assert(n_weights() == in.size());
 	handle_error( cudaMemcpy(weights, &in[0], in.size() * sizeof(F), cudaMemcpyHostToDevice));
}


template <typename F>
void FilterBank<F>::fill(F val) {
	vector<F> vals(n_weights());
	::fill<F>(vals, val);
	from_vector(vals);
}

template <>
void FilterBank<double>::draw_filterbank(string filename) {
}

template <>
void FilterBank<float>::draw_filterbank(string filename) {
	vector<float> filters(n_weights());
	copy_gpu_to_cpu(weights, &filters[0], n_weights());

	int n_filters(in_map * out_map * T);
	int sqrt_n_filters(sqrt(n_filters) + 1);
	vector<float> values(sqrt_n_filters * sqrt_n_filters * kw * kh);
	cout << filters.size() << " " << values.size() << " " << sqrt_n_filters << " " << n_filters << endl;

		for (size_t y(0); y < sqrt_n_filters; ++y)
			for (size_t x(0); x < sqrt_n_filters; ++x) {
				int filter_index(y * sqrt_n_filters + x);
				if (filter_index < n_filters)
					for (size_t fy(0); fy < kh; ++fy)
						for (size_t fx(0); fx < kw; ++fx)
							values[((y * kh) + fy) * sqrt_n_filters * kw + (x * kw) + fx] = filters[filter_index * kw * kh + fy * kw + fx];
			}


	write_img1c(filename, sqrt_n_filters * kw, sqrt_n_filters * kh, &values[0]);
}

template <typename F>
Tensor<F> &operator-=(Tensor<F> &in, Tensor<F> const &other) {
	assert(in.size() == other.size());
	add_cuda<F>(other.data, in.data, in.size(), -1);
	return in;
}


template struct Tensor<float>;
template struct TensorSet<float>;
template struct FilterBank<float>;
template Tensor<float> &operator-=<float>(Tensor<float> &in, Tensor<float> const &other);
// template Tensor<float> &operator*=<float>(Tensor<float> &in, float const other);

template struct Tensor<double>;
template struct TensorSet<double>;
template struct FilterBank<double>;
template Tensor<double> &operator-=<double>(Tensor<double> &in, Tensor<double> const &other);
// template Tensor<double> &operator*=<double>(Tensor<double> &in, double const other);
