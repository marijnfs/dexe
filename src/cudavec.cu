#include "dexe/util.h"
#include "dexe/cudavec.h"
#include "dexe/allocator.h"

uint64_t memory_counter = 0;

namespace dexe {


template <typename F>
CudaVec<F>::CudaVec<F>() : data(0), N(0) 
{}

template <typename F>
CudaVec<F>::CudaVec(int n_) : data(0), N(0) { 
	allocate(n_); 
}


template <typename F>
CudaVec<F>::CudaVec(F *data, int n_) : data(data), N(n_), own(false) 
{}

template <typename F>
CudaVec<F>::CudaVec(CudaVec &other) {
    allocate(other.N);
    copy_gpu_to_gpu(other.data, data, N);
}

template <typename F>
CudaVec<F>::~CudaVec() {
    if (own && N) {
        memory_counter -= N;

        get_allocator()->free((uint8_t*)data);
    }
}

template <typename F>
CudaVec<F> &CudaVec<F>::operator=(CudaVec &other) {
    if (N != other.N) {
        allocate(other.N);
    }
    copy_gpu_to_gpu(other.data, data, N);
    return *this;
}

template <typename F>
bool CudaVec<F>::allocated() {
    return data != NULL;
}

template <typename F>
void CudaVec<F>::free() {
    allocate(0);
}

template <typename F>
void CudaVec<F>::zero(int offset) {
    if (N)
    	get_allocator()->zero(reinterpret_cast<uint8_t*>(data + offset), sizeof(F) * (N - offset));
}

template <typename F>
void CudaVec<F>::init_normal(F mean, F std) { dexe::init_normal<F>(data, N, mean, std); }

template <typename F>
void CudaVec<F>::add_normal(F mean, F std) { dexe::add_normal<F>(data, N, mean, std); }

template <typename F>
void CudaVec<F>::share(CudaVec<F> &other) {
    if (N != other.N) {
        throw DexeException("can't share with CudaVec of different size");
    }
    if (own)
		get_allocator()->free((uint8_t*)data);
    own = false;
    data = other.data;
}

template <typename F>
std::vector<F> CudaVec<F>::to_vector() {
    std::vector<F> vec(N);
    handle_error(cudaMemcpy(&vec[0], data, N * sizeof(F), cudaMemcpyDeviceToHost));
    return vec;
}

template <typename F>
void CudaVec<F>::to_ptr(F *target) {
    handle_error(cudaMemcpy(target, data, N * sizeof(F), cudaMemcpyDeviceToHost));
}

template <typename F>
void CudaVec<F>::from_ptr(F const *source) {
    handle_error(cudaMemcpy(data, source, N * sizeof(F), cudaMemcpyHostToDevice));
}

template <typename F>
void CudaVec<F>::from_vector(std::vector<F> const &vec) {
    if (vec.size() != N)
        allocate(vec.size());
    handle_error(cudaMemcpy(data, &vec[0], N * sizeof(F), cudaMemcpyHostToDevice));
}

template <typename F>
F CudaVec<F>::sum() {
    F result(0);
    if (sizeof(F) == sizeof(float))
        handle_error(cublasSasum(Handler::cublas(), N, data, 1, &result));
    else
        handle_error(cublasDasum(Handler::cublas(), N, data, 1, &result));
    return result;
}


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

template <typename F>
void CudaVec<F>::allocate(int newN) {
    if (!own) {
    	N = newN;
    	return;
    }

    if (N != newN) {
        if (N) {
            memory_counter -= N;

            get_allocator()->free((uint8_t*)data);
            data = 0;
        }
        if (newN) {
            memory_counter += newN;

            data = (F*)get_allocator()->allocate(sizeof(F) * newN);
        }
        N = newN;
    }
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

	v[x] = ::abs(v[x]);
}

__global__ void pow_kernel(float *v, int n, float e) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = ::pow(v[x], e);
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

	v[x] = ::abs(v[x]);
}

__global__ void pow_kerneld(double *v, int n, double e) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = ::pow(v[x], e);
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

}
