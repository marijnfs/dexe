#include <cassert>
#include <cstdlib>

#include "tensor.h"
#include "util.h"
#include "handler.h"
#include "img.h"

using namespace std;

TensorShape::TensorShape(int n, int c, int h, int w) 
	: dimensions{n, c, h, w} {
}

TensorShape::TensorShape(int n, int c, int d, int h, int w)
	: dimensions{n, c, d, h, w} {
}

TensorShape::TensorShape(std::vector<int> dimensions_) 
	: dimensions(dimensions_) {
}

int TensorShape::offset(int n_, int c_, int y_, int x_) {
  return n_ * (c() * w() * h()) + c_ * (w() * h()) + y_ * w() + x_;
}

int TensorShape::n_elements() {
	return calculate_product(dimensions);
}
int TensorShape::n_dimensions() {
	return dimensions.size();
}

void TensorShape::set_c(int c) {
	dimensions[1] = c;
}

int TensorShape::n() {
	return dimensions[0];
}

int TensorShape::c() {
	return dimensions[1];
}

int TensorShape::d() {
	if (dimensions.size() == 5)
		return dimensions[2];
	return 1;
}
int TensorShape::h() {
	if (dimensions.size() == 5)
		return dimensions[3];
	return dimensions[2];
}

int TensorShape::w() {
	if (dimensions.size() == 5)
		return dimensions[4];
	return dimensions[3];
} 

bool TensorShape::operator==(TensorShape const &other) const {
	return dimensions == other.dimensions; 
}

bool TensorShape::operator!=(TensorShape const &other) const {
	return !(*this == other); 
}

template <typename F>
Tensor<F>::Tensor() 
: owning(true) {
	handle_error( cudnnCreateTensorDescriptor(&td) );
}


template <typename F>
Tensor<F>::Tensor(TensorShape s):
  shape(s), owning(true)
{
	handle_error( cudnnCreateTensorDescriptor(&td) );
	allocate();
}


template <typename F>
void Tensor<F>::set_descriptor() {
	vector<int> strides;
	strides.reserve(shape.n_dimensions());

	int stride(1);
	for (auto d_it = shape.dimensions.rbegin(); d_it != shape.dimensions.rend(); ++d_it) {
		strides.emplace_back(stride);
		stride *= *d_it;
	}
	reverse(strides.begin(), strides.end());

	cout << "Shape: " << shape << " strides:" << strides << endl;
	set_descriptor_typed(shape.n_dimensions(), shape.dimensions, strides);
}

template <>
void Tensor<double>::set_descriptor_typed(int N, vector<int> dimensions, vector<int> strides) {
	if (!N) return;
	handle_error( cudnnSetTensorNdDescriptor(td, CUDNN_DATA_DOUBLE, N, dimensions.data(), strides.data()) );	
}

template <>
void Tensor<float>::set_descriptor_typed(int N, vector<int> dimensions, vector<int> strides) {
	cout << "tensor desc, dims: " << dimensions << " strides: " << strides << endl;
	if (!N) return;
	handle_error( cudnnSetTensorNdDescriptor(td, CUDNN_DATA_FLOAT, N, dimensions.data(), strides.data()) );
}

template <typename F>
void Tensor<F>::allocate() {
	if (!owning)
		return;

	if (shape.n_elements() != 0) {
		set_descriptor();
		handle_error( cudaMalloc( (void**)&data, sizeof(F) * shape.n_elements()));
		if (ZERO_ON_INIT)
		  zero();
	}
}


template <typename F>
void Tensor<F>::reshape(TensorShape new_shape) {
	if (!owning) {
		cerr << "Can't reshape non-owning tensor" << endl;
	}

	if (new_shape == shape)
		return;

	//If sizes match but shapes don't, we don't want to deallocate
	if (new_shape.n_elements() != shape.n_elements())
	{
		if (data) {
			cudaFree(data);
			data = nullptr;
		}
		shape = new_shape;
		allocate();
	}
	shape = new_shape;
	set_descriptor();
}


template <typename F>
Tensor<F>::~Tensor() {
	handle_error( cudnnDestroyTensorDescriptor(td));
	if (owning)
	  cudaFree(data);
}


template <typename F>
void Tensor<F>::zero() {
	handle_error( cudaMemset(data, 0, sizeof(F) * size()));
}


template <typename F>
vector<F> Tensor<F>::to_vector() {
	vector<F> vec(shape.n_elements());
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
int Tensor<F>::size() {
	return shape.n_elements();
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
TensorSet<F>::TensorSet(TensorShape shape_) {
	alloc_x(shape_);
}

template <typename F>
TensorSet<F>::TensorSet() {
	alloc_x(TensorShape());
}

template <typename F>
TensorShape TensorSet<F>::shape() {
	if (!x)
		return TensorShape();
	return x->shape;
}

template <typename F>
void TensorSet<F>::alloc_x(TensorShape shape) {
	if (x)
		x->reshape(shape);
	else
		x.reset(new Tensor<F>(shape));
}

template <typename F>
void TensorSet<F>::alloc_grad() {
	if (grad)
		grad->reshape(shape());
	else
		grad.reset(new Tensor<F>(shape()));
}

template <>
FilterBank<float>::FilterBank(std::vector<int> dimensions_) 
	: dimensions(dimensions_) {
	handle_error( cudnnCreateFilterDescriptor(&fd));
	handle_error( cudnnSetFilterNdDescriptor(fd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dimensions.size(), dimensions.data()) );

	handle_error( cudaMalloc( (void**)&weights, sizeof(float) * n_weights()) );
	if (ZERO_ON_INIT)
	  zero();
}

template <>
FilterBank<double>::FilterBank(std::vector<int> dimensions_) 
	: dimensions(dimensions_) {
	handle_error( cudnnCreateFilterDescriptor(&fd));
	handle_error( cudnnSetFilterNdDescriptor(fd, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, dimensions.size(), dimensions.data()) );

	handle_error( cudaMalloc( (void**)&weights, sizeof(double) * n_weights()) );
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
	// cout << "zero: " << in_c << " " << out_c << " " << kw << " " << kh << " " << N << " " << T  << " " << n_weights() << " " << weights << endl;
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
int FilterBank<F>::in_c() {
	return dimensions[0];
}

template <typename F>
int FilterBank<F>::out_c() {
	return dimensions[1];
}

template <typename F>
int FilterBank<F>::kd() {
	if (dimensions.size() == 5)
		return dimensions[2];
	return 1;
}

template <typename F>
int FilterBank<F>::kh() {
	if (dimensions.size() == 5)
		return dimensions[3];
	return dimensions[2];
}

template <typename F>
int FilterBank<F>::kw() {
	if (dimensions.size() == 5)
		return dimensions[4];
	return dimensions[3];
}

template <typename F>
void FilterBank<F>::fill(F val) {
	vector<F> vals(n_weights());
	::fill<F>(vals, val);
	from_vector(vals);
}


template <typename F>
Tensor<F> &operator-=(Tensor<F> &in, Tensor<F> &other) {
	assert(in.size() == other.size());
	add_cuda<F>(other.data, in.data, in.size(), -1);
	return in;
}


template struct Tensor<float>;
template struct TensorSet<float>;
template struct FilterBank<float>;
template Tensor<float> &operator-=<float>(Tensor<float> &in, Tensor<float> &other);
// template Tensor<float> &operator*=<float>(Tensor<float> &in, float const other);

template struct Tensor<double>;
template struct TensorSet<double>;
template struct FilterBank<double>;
template Tensor<double> &operator-=<double>(Tensor<double> &in, Tensor<double> &other);
// template Tensor<double> &operator*=<double>(Tensor<double> &in, double const other);
