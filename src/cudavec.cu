#include "util.h"
#include "cudavec.h"

// template <typename F>
// void CudaVec<F>::rand_zero(F p) {
// 	::rand_zero<F>(data, N, p);
// }

/// Float versions
template <>
CudaVec<float> &CudaVec<float>::sqrt() {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	sqrt_kernel<<<dimGrid, dimBlock>>>(data, N);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<float> &CudaVec<float>::clip(float limit) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	clip_kernel<<<dimGrid, dimBlock>>>(data, N, limit);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<float> &CudaVec<float>::abs() {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

		abs_kernel<<<dimGrid, dimBlock>>>(data, N);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<float> &CudaVec<float>::pow(float e) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	pow_kernel<<<dimGrid, dimBlock>>>(data, N, e);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<float> &CudaVec<float>::exp() {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	exp_kernel<<<dimGrid, dimBlock>>>(data, N);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<float> &CudaVec<float>::add(int idx, float val) {
	add_scalar<<<1, 1>>>(data+idx, val, 1);
		
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<float> &CudaVec<float>::operator+=(float v) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	add_scalar<<<dimGrid, dimBlock>>>(data, v, N);
	
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<float> &CudaVec<float>::operator*=(CudaVec<float> &other) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	times_kernel<<<dimGrid, dimBlock>>>(data, other.data, N);
	
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<float> &CudaVec<float>::operator/=(CudaVec<float> &other) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	divide_kernel<<<dimGrid, dimBlock>>>(data, other.data, N);

	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <typename F>
CudaVec<F> &CudaVec<F>::operator+=(CudaVec<F> &other) {
	add_cuda<F>(other.data, data, N, 1);
	return *this;
}

template <typename F>
CudaVec<F> &CudaVec<F>::operator-=(CudaVec<F> &other) {
	add_cuda<F>(other.data, data, N, -1);
	return *this;
}


template <>
CudaVec<float> &CudaVec<float>::operator*=(float v) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	times_scalar<<<dimGrid, dimBlock>>>(data, v, N);

	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<double> &CudaVec<double>::operator*=(double v) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	times_scalard<<<dimGrid, dimBlock>>>(data, v, N);

	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <typename F>
CudaVec<F> &CudaVec<F>::operator/=(F v) {
	return (*this) *= (1.0 / v);
}

///////Double versions
template <>
CudaVec<double> &CudaVec<double>::sqrt() {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	
		sqrt_kerneld<<<dimGrid, dimBlock>>>(data, N);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<double> &CudaVec<double>::clip(double limit) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	
	clip_kerneld<<<dimGrid, dimBlock>>>(data, N, limit);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<double> &CudaVec<double>::abs() {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	
		abs_kerneld<<<dimGrid, dimBlock>>>(data, N);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<double> &CudaVec<double>::pow(double e) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	
	pow_kerneld<<<dimGrid, dimBlock>>>(data, N, e);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<double> &CudaVec<double>::exp() {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	
		exp_kerneld<<<dimGrid, dimBlock>>>(data, N);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<double> &CudaVec<double>::add(int idx, double val) {
	add_scalard<<<1, 1>>>(data+idx, val, 1);
		
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}


template <>
CudaVec<double> &CudaVec<double>::operator*=(CudaVec<double> &other) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	
		times_kerneld<<<dimGrid, dimBlock>>>(data, other.data, N);
	
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

template <>
CudaVec<double> &CudaVec<double>::operator/=(CudaVec<double> &other) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	
		divide_kerneld<<<dimGrid, dimBlock>>>(data, other.data, N);

	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}



template <>
CudaVec<double> &CudaVec<double>::operator+=(double v) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (N + BLOCKSIZE - 1) / BLOCKSIZE );

	add_scalard<<<dimGrid, dimBlock>>>(data, v, N);
	
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}
///////////////

template <>
float CudaVec<float>::sum() {
	float result(0);
	handle_error( cublasSasum(Handler::cublas(), N, data, 1, &result) );
	return result;
}

template <>
double CudaVec<double>::sum() {
	double result(0);
	handle_error( cublasDasum(Handler::cublas(), N, data, 1, &result) );
	return result;
}

__global__ void clip_kernel(float *v, int n, float limit) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = (v[x] > limit) ? limit : ((v[x] < -limit) ? -limit : v[x]);
}

__global__ void sqrt_kernel(float *v, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = sqrt(v[x]);
}

__global__ void abs_kernel(float *v, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = abs(v[x]);
}

__global__ void pow_kernel(float *v, int n, float e) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = pow(v[x], e);
}

__global__ void exp_kernel(float *v, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = exp(v[x]);
}

__global__ void times_kernel(float *v, float *other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] *= other[x];
}

__global__ void divide_kernel(float *v, float *other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] /= other[x];
}

__global__ void times_scalar(float *v, float other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] *= other;
}

__global__ void add_scalar(float *v, float other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] += other;
}

///Double versions
__global__ void clip_kerneld(double *v, int n, double limit) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = (v[x] > limit) ? limit : ((v[x] < -limit) ? -limit : v[x]);
}

__global__ void sqrt_kerneld(double *v, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = sqrt(v[x]);
}

__global__ void abs_kerneld(double *v, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = abs(v[x]);
}

__global__ void pow_kerneld(double *v, int n, double e) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = pow(v[x], e);
}

__global__ void exp_kerneld(double *v, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = exp(v[x]);
}

__global__ void times_kerneld(double *v, double *other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] *= other[x];
}

__global__ void divide_kerneld(double *v, double *other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] /= other[x];
}

__global__ void times_scalard(double *v, double other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] *= other;
}

__global__ void add_scalard(double *v, double other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] += other;
}


template struct CudaVec<float>;
template struct CudaVec<double>;
