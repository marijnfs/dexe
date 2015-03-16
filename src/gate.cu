#include "gate.h"

__global__ void gate_kerneld(int N, double const *a, double const *b, double *out) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] = a[i] * b[i];
}

__global__ void gate_kernelf(int N, float const *a, float const *b, float *out) {
	int const i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	out[i] = a[i] * b[i];
}

template <>
void gate<double>(Tensor<double> &a, Tensor<double> &b, Tensor<double> &out) {
	int s = a.size();
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( s / BLOCKSIZE + (s % BLOCKSIZE ? 0 : 1));
	
	gate_kerneld<<<dimGrid, dimBlock>>>(a.size(), a.data, b.data, out.data);
}

template <>
void gate<float>(Tensor<float> &a, Tensor<float> &b, Tensor<float> &out) {
	int s = a.size();
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( s / BLOCKSIZE + (s % BLOCKSIZE ? 0 : 1));
	
	gate_kernelf<<<dimGrid, dimBlock>>>(a.size(), a.data, b.data, out.data);
}
