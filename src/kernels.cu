#include "dexe/kernels.h"

namespace dexe {

__global__ void split_kernelf(size_t const N, size_t const C, size_t const X, size_t const Y, float const *input, float *out) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
		return;

    size_t xdiff = i % 2;
    size_t x = (i % X) / 2;
    size_t ii = i / X;
    size_t ydiff = ii % 2;
    size_t y = (ii % Y) / 2;
    ii /= Y;
    
	out[ii * X * Y + (ydiff * 2 + xdiff) * (X * Y / 4) + y * X / 2 + x] = input[i];
}

__global__ void split_kerneld(size_t const N, size_t const C, size_t const X, size_t const Y, double const *input, double *out) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
		return;

    size_t xdiff = i % 2;
    size_t x = (i % X) / 2;
    size_t ii = i / X;
    size_t ydiff = ii % 2;
    size_t y = (ii % Y) / 2;
    ii /= Y;
    
	out[ii * X * Y + (ydiff * 2 + xdiff) * (X * Y / 4) + y * X / 2 + x] = input[i];
}

void split(Tensor<float> &a, Tensor<float> &out) {
	size_t s = a.size();
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (s + BLOCKSIZE - 1) / BLOCKSIZE );

    auto shape = a.shape;
    split_kernelf<<<dimGrid, dimBlock>>>(s, shape.c(), shape.w(), shape.h(), a.ptr(), out.ptr());
}

void split(Tensor<double> &a, Tensor<double> &out) {
	size_t s = a.size();
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (s + BLOCKSIZE - 1) / BLOCKSIZE );

    auto shape = a.shape;
    split_kerneld<<<dimGrid, dimBlock>>>(s, shape.c(), shape.w(), shape.h(), a.ptr(), out.ptr());
}

__global__ void merge_kernelf(size_t const N, size_t const C, size_t const X, size_t const Y, float const *input, float *out) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
		return;

    size_t x = i % X;
    size_t y = (i / X) % Y;
    size_t c = i / X / Y;
    size_t xdiff = c % 2;
    size_t ydiff = (c / 2) % 2;
    size_t cc = c / 4;
    out[cc * X * Y * 4 + (y * 2 + ydiff) * X * 2 + x * 2 + xdiff] = input[i];
}

__global__ void merge_kerneld(size_t const N, size_t const C, size_t const X, size_t const Y, double const *input, double *out) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
		return;

    size_t x = i % X;
    size_t y = (i / X) % Y;
    size_t c = i / X / Y;
    size_t xdiff = c % 2;
    size_t ydiff = (c / 2) % 2;
    size_t cc = c / 4;
    out[cc * X * Y * 4 + (y * 2 + ydiff) * X * 2 + x * 2 + xdiff] = input[i];
}

void merge(Tensor<float> &a, Tensor<float> &out) {
	size_t s = a.size();
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (s + BLOCKSIZE - 1) / BLOCKSIZE );

    auto shape = a.shape;
    merge_kernelf<<<dimGrid, dimBlock>>>(s, shape.c(), shape.w(), shape.h(), a.ptr(), out.ptr());
}


void merge(Tensor<double> &a, Tensor<double> &out) {
	size_t s = a.size();
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (s + BLOCKSIZE - 1) / BLOCKSIZE );

    auto shape = a.shape;
    merge_kerneld<<<dimGrid, dimBlock>>>(s, shape.c(), shape.w(), shape.h(), a.ptr(), out.ptr());
}

__global__ void gate_kerneld(size_t N, double const *a, double const *b, double *out) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] += a[i] * b[i];
}

__global__ void gate_kernelf(size_t N, float const *a, float const *b, float *out) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] += a[i] * b[i];
}


template <>
void gate<double>(Tensor<double> &a, Tensor<double> &b, Tensor<double> &out) {
	size_t s = a.size();
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (s + BLOCKSIZE - 1) / BLOCKSIZE );

	gate_kerneld<<<dimGrid, dimBlock>>>(s, a.ptr(), b.ptr(), out.ptr());
}

template <>
void gate<float>(Tensor<float> &a, Tensor<float> &b, Tensor<float> &out) {
	size_t s = a.size();
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (s  + BLOCKSIZE - 1) / BLOCKSIZE);

	gate_kernelf<<<dimGrid, dimBlock>>>(s, a.ptr(), b.ptr(), out.ptr());
}


////Inverse Gate
__global__ void gateinv_kerneld(size_t N, double const *a, double const *b, double *out) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] += a[i] * (1.0 - b[i]);
}

__global__ void gateinv_kernelf(size_t N, float const *a, float const *b, float *out) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] += a[i] * (1.0 - b[i]);
}

template <>
void gateinv<double>(Tensor<double> &a, Tensor<double> &b, Tensor<double> &out) {
	size_t s = a.size();
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (s + BLOCKSIZE - 1) / BLOCKSIZE );

	gate_kerneld<<<dimGrid, dimBlock>>>(s, a.ptr(), b.ptr(), out.ptr());
}

template <>
void gateinv<float>(Tensor<float> &a, Tensor<float> &b, Tensor<float> &out) {
	size_t s = a.size();
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (s  + BLOCKSIZE - 1) / BLOCKSIZE);

	gate_kernelf<<<dimGrid, dimBlock>>>(s, a.ptr(), b.ptr(), out.ptr());
}


///range

__global__ void range_kerneld(double *a, size_t N, double const min, double const max) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	a[i] = a[i] * (max - min) + min;
}

__global__ void range_kernelf(float *a, size_t N, float const min, float const max) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	a[i] = 	a[i] * (max - min) + min;
}


template <>
void range<float>(float *a, size_t N, float const min, float const max) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	range_kernelf<<<dimGrid, dimBlock>>>(a, N, min, max);
}

template <>
void range<double>(double *a, size_t N, double const min, double const max) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	range_kerneld<<<dimGrid, dimBlock>>>(a, N, min, max);
}


////TANH

__global__ void tanh_forward_kernelf(float *in, float *out, size_t N, float beta, float scale) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;

	//out[i] = in[i];

	out[i] = beta * out[i] + scale * tanh(in[i]);
}


__global__ void tanh_forward_kerneld(double *in, double *out, size_t N, double beta, double scale) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] = beta * out[i] + scale * tanh(in[i]);
}


template <>
void tanh_forward<float>(float *in, float *out, size_t n, float beta, float scale) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	tanh_forward_kernelf<<<dimGrid, dimBlock>>>(in, out, n, beta, scale);
}

template <>
void tanh_forward<double>(double *in, double *out, size_t n, double beta, double scale) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	tanh_forward_kerneld<<<dimGrid, dimBlock>>>(in, out, n, beta, scale);
}

///TANH DERIV
__global__ void tanh_deriv_kernelf(float *out_err, float *act, float *in_err, size_t N, float beta, float scale) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;

	float a = act[i] / scale;
	in_err[i] = beta * in_err[i] + scale * (1.0 - (a * a)) * out_err[i];
}

__global__ void tanh_deriv_kerneld(double *out_err, double *act, double *in_err, size_t N, double beta, double scale) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	double a = act[i] / scale;
	in_err[i] = beta * in_err[i] + scale * (1.0 - (a * a)) * out_err[i];
}

template <>
void tanh_deriv<float>(float *out_err, float *act, float *in_err, size_t n, float beta, float scale) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	tanh_deriv_kernelf<<<dimGrid, dimBlock>>>(out_err, act, in_err, n, beta, scale);
}

template <>
void tanh_deriv<double>(double *out_err, double *act, double *in_err, size_t n, double beta, double scale) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	tanh_deriv_kerneld<<<dimGrid, dimBlock>>>(out_err, act, in_err, n, beta, scale);
}

//SIGMOID FORWARD

__global__ void sigm_forward_kernelf(float *in, float *out, size_t N, float beta, float scale) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	//out[i] = 1;
	out[i] = beta * out[i] + scale / (1.0 + expf(-in[i]));
}


__global__ void sigm_forward_kerneld(double *in, double *out, size_t N, double beta, double scale) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] = beta * out[i] + scale / (1.0 + exp(-in[i]));
}


template <>
void sigm_forward<float>(float *in, float *out, size_t n, float beta, float scale) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	sigm_forward_kernelf<<<dimGrid, dimBlock>>>(in, out, n, beta, scale);
}

template <>
void sigm_forward<double>(double *in, double *out, size_t n, double beta, double scale) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	sigm_forward_kerneld<<<dimGrid, dimBlock>>>(in, out, n, beta, scale);
}


//SIGMOID DERIV
__global__ void sigm_deriv_kernelf(float *out_err, float *act, float *in_err, size_t N, float beta, float scale) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	float a = act[i] / scale;
	in_err[i] = beta * in_err[i] + scale * (1.0 - a) * a * out_err[i];
}

__global__ void sigm_deriv_kerneld(double *out_err, double *act, double *in_err, size_t N, double beta, double scale) {
	size_t const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	double a = act[i] / scale;
	in_err[i] = beta * in_err[i] + scale * (1.0 - a) * a * out_err[i];
}


template <>
void sigm_deriv<float>(float *out_err, float *act, float *in_err, size_t n, float beta, float scale) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	sigm_deriv_kernelf<<<dimGrid, dimBlock>>>(out_err, act, in_err, n, beta, scale);
}

template <>
void sigm_deriv<double>(double *out_err, double *act, double *in_err, size_t n, double beta, double scale) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	sigm_deriv_kerneld<<<dimGrid, dimBlock>>>(out_err, act, in_err, n, beta, scale);
}

template <typename F>
__device__ F device_max(F a, F b) {
	if (a > b)
		return a;
	return b;
}
//support kernel
//assumes one-hot encoding with 1 on and 0 off.
//could be updated to have more efficient encoding
template <typename F>
__global__ void support_kernel(F *prediction, F *target, F *loss, size_t N, F support) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N)
		return;

	if (target[i] > 0.5)
		loss[i] = device_max(F(0.0), support - prediction[i]);
	else
		loss[i] = -device_max(F(0.0), prediction[i] + support);
}

template <typename F>
void support_loss(F *input, F *target, F *loss, size_t N, F support) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE);

	support_kernel<<<dimGrid, dimBlock>>>(input, target, loss, N, support);
}


/// Dice kernel
template <typename F>
__global__ void dice_kernel(F *prediction, F *target, F *loss, F *conjunction_sum, F *disjunction_sum, size_t N) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N)
		return;
#ifdef __CUDA_ARCH__
    #if (__CUDA_ARCH__ >= 600)
	if (target[i])
		atomicAdd(conjunction_sum + blockIdx.x, target[i] * prediction[i]);	
	atomicAdd(disjunction_sum + blockIdx.x, target[i] + prediction[i]);
    #endif
#endif
	loss[i] = target[i] - prediction[i];	
}

template <typename F>
void dice_loss(F *input, F *target, F *loss, F *cpu_conjunction, F *cpu_disjunction, size_t N) {
#ifdef __CUDA_ARCH__
    #if (__CUDA_ARCH__ < 600)
		std::cerr << "Warning Dice loss can't be used on this cuda architecture" << std::endl;
	#endif
#endif
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t n_blocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
	size_t dimGrid( n_blocks );

	CudaVec<F> tmp_conjunction_sum(n_blocks);
	CudaVec<F> tmp_disjuntion_sum(n_blocks);

	tmp_conjunction_sum.zero();
	tmp_disjuntion_sum.zero();

	dice_kernel<<<dimGrid, dimBlock>>>(input, target, loss, tmp_conjunction_sum.data, tmp_disjuntion_sum.data, N);

	*cpu_conjunction = tmp_conjunction_sum.sum();
	*cpu_disjunction = tmp_disjuntion_sum.sum();
}


/// In-place threshold kernel
/// sets values above threshold to 1, 0 otherwise
template <typename F>
__global__ void threshold_kernel(F *input, size_t N, F threshold) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N)
		return;

	if (input[i] > threshold)
		input[i] = 1.0;
	else
		input[i] = 0.0;
}

template <typename F>
void threshold_cuda(F *input, size_t N, F threshold) {
	size_t const BLOCKSIZE(1024);

	size_t dimBlock( BLOCKSIZE );
	size_t dimGrid( (N  + BLOCKSIZE - 1) / BLOCKSIZE);

	threshold_kernel<<<dimGrid, dimBlock>>>(input, N, threshold);
}




template void support_loss<float>(float *input, float *target, float *loss, size_t N, float support);
template void support_loss<double>(double *input, double *target, double *loss, size_t N, double support);

template void dice_loss<float>(float *input, float *target, float *loss, float *cpu_conjunction, float *cpu_disjunction, size_t N);
template void dice_loss<double>(double *input, double *target, double *loss, double *cpu_conjunction, double *cpu_disjunction, size_t N);


template void threshold_cuda<float>(float *input, size_t N, float threshold);
template void threshold_cuda<double>(double *input, size_t N, double threshold);

}