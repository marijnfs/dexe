#include <cassert>
#include <cstdlib>

#include <cereal/cereal.hpp>

#include "tensor.h"
#include "util.h"
#include "handler.h"
#include "img.h"
#include "kernels.h"
#include "cudavec.h"


using namespace std;

namespace dexe {

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
	if (dimensions.empty())
		return 0;
	return calculate_product(dimensions);
}

int TensorShape::n_dimensions() {
	return dimensions.size();
}

int TensorShape::n_pixels() {
	if (dimensions.size() <= 2)
		return 1;
	std::vector<int> pixelDims(dimensions.begin() + 2, dimensions.end());
	return calculate_product(pixelDims);
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

int &TensorShape::operator[](int index) {
    return dimensions[index];
}

template <typename F>
Tensor<F>::Tensor() 
: owning(true) {
	handle_error( cudnnCreateTensorDescriptor(&td) );
}


template <typename F>
Tensor<F>::Tensor(TensorShape s, cudnnTensorFormat_t format_):
  shape(s), owning(true), format(format_) {
	handle_error( cudnnCreateTensorDescriptor(&td) );
	allocate();
	set_descriptor();
}

template <typename F>
Tensor<F>::Tensor(TensorShape s, F *data_, cudnnTensorFormat_t format_):
  shape(s), owning(false), data(data_), format(format_) {
	handle_error( cudnnCreateTensorDescriptor(&td) );
	allocate();
}


template <typename F>
Tensor<F>::Tensor(std::vector<F> data) : shape(TensorShape({1, 1, (int)data.size()})), owning(true) {
	handle_error( cudnnCreateTensorDescriptor(&td) );
	allocate();
	from_vector(data);
}

template <typename F>
void Tensor<F>::set_descriptor() {
	set_descriptor_typed();
}

template <>
void Tensor<float>::set_descriptor_typed() {
	if (!shape.n_dimensions() || !shape.n_elements()) return;
	handle_error( cudnnSetTensorNdDescriptorEx(td, format, CUDNN_DATA_FLOAT, shape.n_dimensions(), shape.dimensions.data()) );
}

template <>
void Tensor<double>::set_descriptor_typed() {
	if (!shape.n_dimensions() || !shape.n_elements()) return;
	handle_error( cudnnSetTensorNdDescriptorEx(td, format, CUDNN_DATA_DOUBLE, shape.n_dimensions(), shape.dimensions.data()) );
}

template <typename F>
void Tensor<F>::allocate() {
	if (!owning)
		return;

	if (shape.n_elements() != 0) {
		cout << "allocating: " << shape << " " << shape.n_elements() << endl;
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

	//new shape is the same, no need to do anything
	if (new_shape == shape)
		return;

	//If sizes match but shapes don't, we don't want to deallocate
	cout << "Reshape : " << shape << " to " << new_shape << " elements: " << shape.n_elements() << " " << new_shape.n_elements() << endl;
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
void Tensor<F>::threshold(F value) {
	threshold_cuda(data, size(), value);
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
		throw DexeException("sizes don't match");
	}
 	handle_error( cudaMemcpy(data, &in[0], in.size() * sizeof(F), cudaMemcpyHostToDevice));
}

template <typename F>
void Tensor<F>::from_tensor(Tensor<F> &in, F alpha) {
	if (size() != in.size()) {
 			throw DexeException("sizes don't match");
	}
	F beta(0);
	handle_error( cudnnTransformTensor(Handler::cudnn(), &alpha, in.td, in.data, &beta, td, data) );
}

template <typename F>
void Tensor<F>::from_ptr(F const *in) {
	handle_error( cudaMemcpy(data, in, size() * sizeof(F), cudaMemcpyHostToDevice));
}

template <typename F>
void Tensor<F>::init_normal(F mean, F std) {
	//size_t even_size(((size() + 1) / 2) * 2);
	dexe::init_normal(data, size(), mean, std);
	// size_t even_size(size());
	// handle_error( curandGenerateNormal(Handler::curand(), data, even_size, mean, std) );
}


template <typename F>
void Tensor<F>::init_uniform(F var) {
  dexe::init_uniform(data, size(), var);
  /*
	vector<F> vec = to_vector();
	for (size_t i(0); i < vec.size(); ++i)
		vec[i] = -var + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)/(2.0 * var));
	from_vector(vec);
  */
}

template <typename F>
void Tensor<F>::add(Tensor<F> &in, F alpha) {
	if (size() != in.size()) {
		throw DexeException("sizes don't match");
	}
	F beta(1);
	handle_error( cudnnTransformTensor(Handler::cudnn(), &alpha, in.td, in.data, &beta, td, data) );
}

template <typename F>
void Tensor<F>::scale(F alpha) {
	scale_cuda(data, shape.n_elements(), alpha);
}

template <typename F>
void Tensor<F>::fill(F val) {
	vector<F> vals(size());
	dexe::fill<F>(vals, val);
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
float Tensor<float>::asum() {
	float result(0);
	handle_error( cublasSasum(Handler::cublas(), size(), data, 1, &result) );
	return result;
}

template <>
double Tensor<double>::asum() {
	double result(0);
	handle_error( cublasDasum(Handler::cublas(), size(), data, 1, &result) );
	return result;
}

template <>
float Tensor<float>::sum() {
	static Tensor<float> one({1});
	float result(0);
	handle_error( cublasSdot(Handler::cublas(), size(), data, 1, one.data, 0, &result) );
	return result;
}

template <typename F>
F Tensor<F>::mean() {
	return sum() / shape.n_elements();
}

template <>
double Tensor<double>::sum() {
	Tensor<double> one({1});
	double result(0);
	handle_error( cublasDdot(Handler::cublas(), size(), data, 1, one.data, 0, &result) );
	return result;
}

template <typename F>
Tensor<F> &Tensor<F>::operator*=(F val) {
	CudaVec<F> vec(data, size());
	vec *= val;
	return *this;
}

template <typename F>
Tensor<F> &Tensor<F>::operator/=(F val) {
	CudaVec<F> vec(data, size());
	vec /= val;
	return *this;
}

template <typename F>
Tensor<F> &Tensor<F>::operator-=(F val) {
	CudaVec<F> vec(data, size());
	vec += -val;
	return *this;
}

template <typename F>
Tensor<F> &Tensor<F>::operator+=(Tensor<F> &t) {
	CudaVec<F> vec(data, size());
	CudaVec<F> other(t.data, t.size());
	vec += other;
	return *this;
}

template <typename F>
Tensor<F> &Tensor<F>::operator*=(Tensor<F> &t) {
	CudaVec<F> vec(data, size());
	CudaVec<F> other(t.data, t.size());
	vec *= other;
	return *this;
}

template <typename F>
Tensor<F> &Tensor<F>::operator-=(Tensor<F> &t) {
	CudaVec<F> vec(data, size());
	CudaVec<F> other(t.data, t.size());
	vec -= other;
	return *this;
}

template <typename F>
TensorSet<F>::TensorSet(TensorShape shape_) {
	alloc_x(shape_);
	alloc_grad(TensorShape()); //always start with empty grad, we might just be doing forward
}

template <typename F>
TensorSet<F>::TensorSet() {
	alloc_x(TensorShape());
	alloc_grad(TensorShape());
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
void TensorSet<F>::alloc_grad(TensorShape shape) {
	if (grad)
		grad->reshape(shape);
	else
		grad.reset(new Tensor<F>(shape));
}

template <typename F>
FilterBank<F>::FilterBank() {

}

template <typename F>
FilterBank<F>::FilterBank(std::vector<int> dimensions_) 
	: dimensions(dimensions_) {
	init();
}



template <typename F>
FilterBank<F>::~FilterBank() {
	cudnnDestroyFilterDescriptor(fd);
	cudaFree(weights);
}

template <typename F>
void FilterBank<F>::reshape(std::vector<int> dimensions_) {
	dimensions = dimensions_;
	init();
}

template <typename F>
void FilterBank<F>::init() {
	if (fd)
		handle_error( cudnnDestroyFilterDescriptor(fd));

    auto dimensions_copy = dimensions;
    // if (DEFAULT_TENSOR_FORMAT == CUDNN_TENSOR_NHWC) { //channel last filters have the input channels last, so we adjust
    //     auto in_c = dimensions_copy[1];
    //     dimensions_copy[1] = dimensions_copy.back();
    //     dimensions_copy.back() = in_c;
    // }
    
	handle_error( cudnnCreateFilterDescriptor(&fd));
	handle_error( cudnnSetFilterNdDescriptor(fd, (sizeof(F) == sizeof(float)) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE, DEFAULT_TENSOR_FORMAT, dimensions_copy.size(), dimensions_copy.data()) );

	if (weights)
		handle_error( cudaFree(weights) );

	handle_error( cudaMalloc( (void**)&weights, sizeof(F) * n_weights()) );
	if (ZERO_ON_INIT)
	  zero();		
}

template <typename F>
void FilterBank<F>::init_normal(F mean, F std) {
	dexe::init_normal(weights, n_weights(), mean, std);
}

template <typename F>
void FilterBank<F>::init_uniform(F var) {
	dexe::init_uniform(weights, n_weights(), var);
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
int FilterBank<F>::out_c() {
	return dimensions[0];
}

template <typename F>
int FilterBank<F>::in_c() {
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
	dexe::fill<F>(vals, val);
	from_vector(vals);
}


// template <typename F>
// Tensor<F> &operator-=(Tensor<F> &in, Tensor<F> &other) {
// 	assert(in.size() == other.size());
// 	add_cuda<F>(other.data, in.data, in.size(), -1);
// 	return in;
// }


template struct Tensor<float>;
template struct TensorSet<float>;
template struct FilterBank<float>;
// template Tensor<float> &operator-=<float>(Tensor<float> &in, Tensor<float> &other);
// template Tensor<float> &operator*=<float>(Tensor<float> &in, float const other);

template struct Tensor<double>;
template struct TensorSet<double>;
template struct FilterBank<double>;
// template Tensor<double> &operator-=<double>(Tensor<double> &in, Tensor<double> &other);
// template Tensor<double> &operator*=<double>(Tensor<double> &in, double const other);

}
