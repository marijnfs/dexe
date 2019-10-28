#include "kernels.h"

__global__ void split_kernelf(int const N, int const C, int const X, int const Y, float const *input, float *out) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
		return;

    int xdiff = i % 2;
    int x = (i % X) / 2;
    int ii = i / X;
    int ydiff = ii % 2;
    int y = (ii % Y) / 2;
    ii /= Y;
    
	out[ii * X * Y + (ydiff * 2 + xdiff) * (X * Y / 4) + y * X / 2 + x] = input[i];
    
}

void split(Tensor<float> &a, Tensor<float> &out) {
	int s = a.size();
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (s + BLOCKSIZE - 1) / BLOCKSIZE );

    auto shape = a.shape;
    split_kernelf<<<dimGrid, dimBlock>>>(s, shape.c(), shape.w(), shape.h(), a.data, out.data);
}

__global__ void merge_kernelf(int const N, int const C, int const X, int const Y, float const *input, float *out) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
		return;

    int x = i % X;
    int y = (i / X) % Y;
    int c = i / X / Y;
    int xdiff = c % 2;
    int ydiff = (c / 2) % 2;
    int cc = c / 4;
    out[cc * X * Y * 4 + (y * 2 + ydiff) * X * 2 + x * 2 + xdiff] = input[i];
}

void merge(Tensor<float> &a, Tensor<float> &out) {
	int s = a.size();
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (s + BLOCKSIZE - 1) / BLOCKSIZE );

    auto shape = a.shape;
    merge_kernelf<<<dimGrid, dimBlock>>>(s, shape.c(), shape.w(), shape.h(), a.data, out.data);
}


__global__ void gate_kerneld(int N, double const *a, double const *b, double *out) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] += a[i] * b[i];
}

__global__ void gate_kernelf(int N, float const *a, float const *b, float *out) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] += a[i] * b[i];
}


template <>
void gate<double>(Tensor<double> &a, Tensor<double> &b, Tensor<double> &out) {
	int s = a.size();
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (s + BLOCKSIZE - 1) / BLOCKSIZE );

	gate_kerneld<<<dimGrid, dimBlock>>>(s, a.data, b.data, out.data);
}

template <>
void gate<float>(Tensor<float> &a, Tensor<float> &b, Tensor<float> &out) {
	int s = a.size();
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (s  + BLOCKSIZE - 1) / BLOCKSIZE);

	gate_kernelf<<<dimGrid, dimBlock>>>(s, a.data, b.data, out.data);
}


////Inverse Gate
__global__ void gateinv_kerneld(int N, double const *a, double const *b, double *out) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] += a[i] * (1.0 - b[i]);
}

__global__ void gateinv_kernelf(int N, float const *a, float const *b, float *out) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] += a[i] * (1.0 - b[i]);
}

template <>
void gateinv<double>(Tensor<double> &a, Tensor<double> &b, Tensor<double> &out) {
	int s = a.size();
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (s + BLOCKSIZE - 1) / BLOCKSIZE );

	gate_kerneld<<<dimGrid, dimBlock>>>(s, a.data, b.data, out.data);
}

template <>
void gateinv<float>(Tensor<float> &a, Tensor<float> &b, Tensor<float> &out) {
	int s = a.size();
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (s  + BLOCKSIZE - 1) / BLOCKSIZE);

	gate_kernelf<<<dimGrid, dimBlock>>>(s, a.data, b.data, out.data);
}


///range

__global__ void range_kerneld(double *a, int N, double const min, double const max) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	a[i] = a[i] * (max - min) + min;
}

__global__ void range_kernelf(float *a, int N, float const min, float const max) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	a[i] = 	a[i] * (max - min) + min;
}


template <>
void range<float>(float *a, int N, float const min, float const max) {
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	range_kernelf<<<dimGrid, dimBlock>>>(a, N, min, max);
}

template <>
void range<double>(double *a, int N, double const min, double const max) {
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	range_kerneld<<<dimGrid, dimBlock>>>(a, N, min, max);
}


////TANH

__global__ void tanh_forward_kernelf(float *in, float *out, int N, float beta, float scale) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;

	//out[i] = in[i];

	out[i] = beta * out[i] + scale * tanh(in[i]);
}


__global__ void tanh_forward_kerneld(double *in, double *out, int N, double beta, double scale) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] = beta * out[i] + scale * tanh(in[i]);
}


template <>
void tanh_forward<float>(float *in, float *out, int n, float beta, float scale) {
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	tanh_forward_kernelf<<<dimGrid, dimBlock>>>(in, out, n, beta, scale);
}

template <>
void tanh_forward<double>(double *in, double *out, int n, double beta, double scale) {
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	tanh_forward_kerneld<<<dimGrid, dimBlock>>>(in, out, n, beta, scale);
}

///TANH DERIV
__global__ void tanh_deriv_kernelf(float *out_err, float *act, float *in_err, int N, float beta, float scale) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;

	float a = act[i] / scale;
	in_err[i] = beta * in_err[i] + scale * (1.0 - (a * a)) * out_err[i];
}

__global__ void tanh_deriv_kerneld(double *out_err, double *act, double *in_err, int N, double beta, double scale) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	double a = act[i] / scale;
	in_err[i] = beta * in_err[i] + scale * (1.0 - (a * a)) * out_err[i];
}

template <>
void tanh_deriv<float>(float *out_err, float *act, float *in_err, int n, float beta, float scale) {
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	tanh_deriv_kernelf<<<dimGrid, dimBlock>>>(out_err, act, in_err, n, beta, scale);
}

template <>
void tanh_deriv<double>(double *out_err, double *act, double *in_err, int n, double beta, double scale) {
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	tanh_deriv_kerneld<<<dimGrid, dimBlock>>>(out_err, act, in_err, n, beta, scale);
}

//SIGMOID FORWARD

__global__ void sigm_forward_kernelf(float *in, float *out, int N, float beta, float scale) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	//out[i] = 1;
	out[i] = beta * out[i] + scale / (1.0 + expf(-in[i]));
}


__global__ void sigm_forward_kerneld(double *in, double *out, int N, double beta, double scale) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] = beta * out[i] + scale / (1.0 + exp(-in[i]));
}


template <>
void sigm_forward<float>(float *in, float *out, int n, float beta, float scale) {
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	sigm_forward_kernelf<<<dimGrid, dimBlock>>>(in, out, n, beta, scale);
}

template <>
void sigm_forward<double>(double *in, double *out, int n, double beta, double scale) {
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	sigm_forward_kerneld<<<dimGrid, dimBlock>>>(in, out, n, beta, scale);
}


//SIGMOID DERIV

__global__ void sigm_deriv_kernelf(float *out_err, float *act, float *in_err, int N, float beta, float scale) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	float a = act[i] / scale;
	in_err[i] = beta * in_err[i] + scale * (1.0 - a) * a * out_err[i];
}

__global__ void sigm_deriv_kerneld(double *out_err, double *act, double *in_err, int N, double beta, double scale) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	double a = act[i] / scale;
	in_err[i] = beta * in_err[i] + scale * (1.0 - a) * a * out_err[i];
}


template <>
void sigm_deriv<float>(float *out_err, float *act, float *in_err, int n, float beta, float scale) {
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	sigm_deriv_kernelf<<<dimGrid, dimBlock>>>(out_err, act, in_err, n, beta, scale);
}

template <>
void sigm_deriv<double>(double *out_err, double *act, double *in_err, int n, double beta, double scale) {
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (n  + BLOCKSIZE - 1) / BLOCKSIZE);

	sigm_deriv_kerneld<<<dimGrid, dimBlock>>>(out_err, act, in_err, n, beta, scale);
}
