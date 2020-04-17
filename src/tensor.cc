#include <cassert>
#include <cstdlib>

#include "cereal/cereal.hpp"

#include "dexe/cudavec.h"
#include "dexe/handler.h"
#include "dexe/kernels.h"
#include "dexe/tensor.h"
#include "dexe/util.h"

using namespace std;

namespace dexe {
TensorShape::TensorShape(int n, int c, int h) : dimensions{n, c, h} {}

TensorShape::TensorShape(int n, int c, int h, int w) : dimensions{n, c, h, w} {}

TensorShape::TensorShape(int n, int c, int d, int h, int w)
    : dimensions{n, c, d, h, w} {}

TensorShape::TensorShape(std::vector<int> dimensions_)
    : dimensions(dimensions_) {}

int TensorShape::offset(int n_, int c_, int y_, int x_) {
    return n_ * (c() * w() * h()) + c_ * (w() * h()) + y_ * w() + x_;
}

int TensorShape::n_elements() {
    if (dimensions.empty())
        return 0;
    return calculate_product(dimensions);
}

int TensorShape::n_dimensions() { return dimensions.size(); }

int TensorShape::n_pixels() {
    if (dimensions.size() <= 2)
        return 1;
    std::vector<int> pixelDims(dimensions.begin() + 2, dimensions.end());
    return calculate_product(pixelDims);
}

void TensorShape::set_c(int c) { dimensions[1] = c; }

int TensorShape::n() { return dimensions[0]; }

int TensorShape::c() { return dimensions[1]; }

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

int &TensorShape::operator[](int index) { return dimensions[index]; }

TensorShape TensorShape::zero_lowerdims() {
    auto dims = dimensions;
    std::fill(dims.begin() + 2, dims.end(), 0);
    return TensorShape(dims);
}

template <typename F> Tensor<F>::Tensor() : owning(true) {
    handle_error(cudnnCreateTensorDescriptor(&td));
}

template <typename F>
Tensor<F>::Tensor(TensorShape s, cudnnTensorFormat_t format_)
    : shape(s), owning(true), format(format_) {
    handle_error(cudnnCreateTensorDescriptor(&td));
    allocate();
    set_descriptor();
}


template <typename F>
std::unique_ptr<Tensor<F>> Tensor<F>::from_vector_data(std::vector<F> &&data) {
    auto t = std::make_unique<Tensor<F>>(TensorShape(1, 1, (int)data.size()));
    t->from_vector(data);
    return t;
}


template <typename F> void Tensor<F>::set_descriptor() {
    set_descriptor_typed();
}

template <> void Tensor<float>::set_descriptor_typed() {
    if (!shape.n_dimensions() || !shape.n_elements())
        return;
    handle_error(cudnnSetTensorNdDescriptorEx(td, format, CUDNN_DATA_FLOAT,
                                              shape.n_dimensions(),
                                              shape.dimensions.data()));
}

template <> void Tensor<double>::set_descriptor_typed() {
    if (!shape.n_dimensions() || !shape.n_elements())
        return;
    handle_error(cudnnSetTensorNdDescriptorEx(td, format, CUDNN_DATA_DOUBLE,
                                              shape.n_dimensions(),
                                              shape.dimensions.data()));
}

template <typename F> void Tensor<F>::allocate() {
    if (!owning)
        return;

    if (shape.n_elements() != 0) {
        cudavec.allocate(shape.n_elements());
        cudavec.zero();
    }
}

template <typename F> bool Tensor<F>::allocated() { return cudavec.allocated(); }

template <typename F> void Tensor<F>::reshape(TensorShape new_shape, bool force_zero) {
    if (!owning) {
        cerr << "Can't reshape non-owning tensor" << endl;
    }

    // new shape is the same, no need to do anything
    if (new_shape == shape)
        return;

    // If sizes match but shapes don't, we don't want to deallocate
    if (new_shape.n_elements() != shape.n_elements()) {
        cudavec.free();
        shape = new_shape;
        allocate();
    }
    shape = new_shape;
    set_descriptor();

    if (force_zero)
        zero();
}

template <typename F> Tensor<F>::~Tensor() {
    handle_error(cudnnDestroyTensorDescriptor(td));
}

template <typename F> void Tensor<F>::zero() {
    cudavec.zero();
}

template <typename F> void Tensor<F>::threshold(F value) {
    threshold_cuda(ptr(), size(), value);
}

template <typename F> vector<F> Tensor<F>::to_vector() {
    return cudavec.to_vector();
}

template <typename F> void Tensor<F>::to_ptr(F *target) {
    cudavec.to_ptr(target);
}

template <typename F> void Tensor<F>::from_vector(vector<F> &in) {
    if (size() != in.size()) {
        throw DexeException("sizes don't match");
    }
    cudavec.from_vector(in);
}

template <typename F> void Tensor<F>::from_tensor(Tensor<F> &in, F alpha) {
    if (size() != in.size()) {
        throw DexeException("sizes don't match");
    }
    F beta(0);
    handle_error(cudnnTransformTensor(Handler::cudnn(), &alpha, in.td, in.ptr(),
                                      &beta, td, ptr()));
}

template <typename F> void Tensor<F>::from_ptr(F const *source) {
    cudavec.from_ptr(source);
}

template <typename F> void Tensor<F>::init_normal(F mean, F std) {
    cudavec.init_normal(mean, std);
}

template <typename F> void Tensor<F>::init_uniform(F var) {
    dexe::init_uniform(ptr(), size(), var);
}

template <typename F> void Tensor<F>::add(Tensor<F> &in, F alpha) {
    if (size() != in.size()) {
        throw DexeException("sizes don't match");
    }
    F beta(1);
    handle_error(cudnnTransformTensor(Handler::cudnn(), &alpha, in.td, in.ptr(),
                                      &beta, td, ptr()));
}

template <typename F> void Tensor<F>::scale(F alpha) {
    scale_cuda(ptr(), shape.n_elements(), alpha);
}

template <typename F> void Tensor<F>::fill(F val) {
    vector<F> vals(size());
    dexe::fill<F>(vals, val);
    from_vector(vals);
}

template <typename F> int Tensor<F>::size() { return shape.n_elements(); }

template <> float Tensor<float>::norm() {
    float result(0);
    handle_error(
        cublasSdot(Handler::cublas(), size(), ptr(), 1, ptr(), 1, &result));
    return sqrt(result);
}

template <> double Tensor<double>::norm() {
    double result(0);
    handle_error(
        cublasDdot(Handler::cublas(), size(), ptr(), 1, ptr(), 1, &result));
    return sqrt(result);
}

template <> float Tensor<float>::norm2() {
    float result(0);
    handle_error(
        cublasSnrm2(Handler::cublas(), size(), ptr(), 1, &result));
    return result * result;
}

template <> double Tensor<double>::norm2() {
    double result(0);
    handle_error(
        cublasDnrm2(Handler::cublas(), size(), ptr(), 1, &result));
    return result * result;
}

template <> float Tensor<float>::asum() {
    float result(0);
    handle_error(cublasSasum(Handler::cublas(), size(), ptr(), 1, &result));
    return result;
}

template <> double Tensor<double>::asum() {
    double result(0);
    handle_error(cublasDasum(Handler::cublas(), size(), ptr(), 1, &result));
    return result;
}

template <> float Tensor<float>::sum() {
    auto one = Handler::one_float();

    float result(0);
    handle_error(
        cublasSdot(Handler::cublas(), size(), ptr(), 1, one, 0, &result));
    return result;
}

template <typename F> F Tensor<F>::mean() { return sum() / shape.n_elements(); }

template <> double Tensor<double>::sum() {
    auto one = Handler::one_double();

    double result(0);
    handle_error(
        cublasDdot(Handler::cublas(), size(), ptr(), 1, one, 0, &result));
    return result;
}

template <typename F> Tensor<F> &Tensor<F>::operator*=(F val) {
    CudaVec<F> vec(ptr(), size());
    vec *= val;
    return *this;
}

template <typename F> Tensor<F> &Tensor<F>::operator/=(F val) {
    CudaVec<F> vec(ptr(), size());
    vec /= val;
    return *this;
}

template <typename F> Tensor<F> &Tensor<F>::operator-=(F val) {
    CudaVec<F> vec(ptr(), size());
    vec += -val;
    return *this;
}

template <typename F> Tensor<F> &Tensor<F>::operator+=(Tensor<F> &t) {
    CudaVec<F> vec(ptr(), size());
    CudaVec<F> other(t.ptr(), t.size());
    vec += other;
    return *this;
}

template <typename F> Tensor<F> &Tensor<F>::operator*=(Tensor<F> &t) {
    CudaVec<F> vec(ptr(), size());
    CudaVec<F> other(t.ptr(), t.size());
    vec *= other;
    return *this;
}

template <typename F> Tensor<F> &Tensor<F>::operator-=(Tensor<F> &t) {
    CudaVec<F> vec(ptr(), size());
    CudaVec<F> other(t.ptr(), t.size());
    vec -= other;
    return *this;
}

template <typename F> TensorSet<F>::TensorSet(TensorShape shape_) {
    alloc_x(shape_);
    alloc_grad(TensorShape()); // always start with empty grad, we might just be
                               // doing forward
}

template <typename F> TensorSet<F>::TensorSet() {
    alloc_x(TensorShape());
    alloc_grad(TensorShape());
}

template <typename F> TensorShape TensorSet<F>::shape() {
    if (!x)
        return TensorShape();
    return x->shape;
}

template <typename F> void TensorSet<F>::alloc_x(TensorShape shape) {
    if (x)
        x->reshape(shape);
    else
        x.reset(new Tensor<F>(shape));
}

template <typename F> void TensorSet<F>::alloc_grad(TensorShape shape) {
    if (grad)
        grad->reshape(shape);
    else
        grad.reset(new Tensor<F>(shape));
}

template <typename F> FilterBank<F>::FilterBank() {}

template <typename F>
FilterBank<F>::FilterBank(std::vector<int> dimensions_)
    : weights(TensorShape(dimensions_)) {
    init_descriptor();
}

template <typename F> FilterBank<F>::~FilterBank() {
    cudnnDestroyFilterDescriptor(fd);
    // handle_error( cudaFree(weights) );
}

template <typename F>
void FilterBank<F>::reshape(std::vector<int> dimensions_) {
    weights.reshape(TensorShape(dimensions_));
    init_descriptor();
}

template <typename F> void FilterBank<F>::init_descriptor() {
    if (fd)
        handle_error(cudnnDestroyFilterDescriptor(fd));

    auto dimensions_copy = dimensions();

    handle_error(cudnnCreateFilterDescriptor(&fd));
    handle_error(cudnnSetFilterNdDescriptor(
        fd, (sizeof(F) == sizeof(float)) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
        DEFAULT_TENSOR_FORMAT, dimensions_copy.size(), dimensions_copy.data()));
}

template <typename F> void FilterBank<F>::init_normal(F mean, F std) {
    weights.init_normal(mean, std);
}

template <typename F> void FilterBank<F>::init_uniform(F var) {
    weights.init_uniform(var);
}

template <typename F> void FilterBank<F>::zero() {
    weights.zero();
}

template <typename F> vector<F> FilterBank<F>::to_vector() {
    return weights.to_vector();
}

template <typename F> void FilterBank<F>::from_vector(vector<F> &in) {
    assert(n_weights() == in.size());
    weights.from_vector(in);
}

template <typename F> int FilterBank<F>::out_c() { return dimensions()[0]; }

template <typename F> int FilterBank<F>::in_c() { return dimensions()[1]; }

template <typename F> int FilterBank<F>::kd() {
    if (dimensions().size() == 5)
        return dimensions()[2];
    return 1;
}

template <typename F> int FilterBank<F>::kh() {
    if (dimensions().size() == 5)
        return dimensions()[3];
    return dimensions()[2];
}

template <typename F> int FilterBank<F>::kw() {
    if (dimensions().size() == 5)
        return dimensions()[4];
    return dimensions()[3];
}

template <typename F> void FilterBank<F>::fill(F val) {
    vector<F> vals(n_weights());
    dexe::fill<F>(vals, val);
    from_vector(vals);
}

template struct Tensor<float>;
template struct TensorSet<float>;
template struct FilterBank<float>;

template struct Tensor<double>;
template struct TensorSet<double>;
template struct FilterBank<double>;

} // namespace dexe
