#include "gate.h"

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