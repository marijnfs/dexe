#include "util.h"
#include "handler.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>

__device__ __forceinline__ int get_index(int X, int Y, int Z, int C, int x, int y, int z) {
  return z * (C * X * Y) + y * X + x;
}

__device__ __forceinline__ void add_c(float const *in, float *out, int slicesizein, int slicesizeout, int C) {
  // *out = *in + *out;
	for (size_t c(0); c < C; ++c)
		out[c * slicesizeout] += in[c * slicesizein];
}

__device__ __forceinline__ void copy_c(float const *in, float *out, int slicesizein, int slicesizeout, int C) {
  // *out = *in;
  for (size_t c(0); c < C; ++c)
   out[c * slicesizeout] = in[c * slicesizein];
}

__global__ void normal_kernel(int seed, float *data, int n, float mean, float std) {
  if (threadIdx.x != 0) return;
  curandState state;

  curand_init(seed, 0, 0, &state);
  for (size_t i(0); i < n; ++i)
    data[i] = curand_normal(&state) * std + mean;
}

__global__ void normal_kerneld(int seed, double *data, int n, double mean, double std) {
  if (threadIdx.x != 0) return;
  curandState state;
  curand_init(seed, 0, 0, &state);
  for (size_t i(0); i < n; ++i)
    data[i] = curand_normal_double(&state) * std + mean;
}

__global__ void add_normal_kernel(int seed, float *data, int n, float mean, float std) {
  if (threadIdx.x != 0) return;
  curandState state;

  curand_init(seed, 0, 0, &state);
  for (size_t i(0); i < n; ++i)
    data[i] += curand_normal(&state) * std + mean;
}

__global__ void add_normal_kerneld(int seed, double *data, int n, double mean, double std) {
  if (threadIdx.x != 0) return;
  curandState state;
  curand_init(seed, 0, 0, &state);
  for (size_t i(0); i < n; ++i)
    data[i] += curand_normal_double(&state) * std + mean;
}

template <>
void init_normal<float>(float *a, int N, float mean, float std) {
     normal_kernel<<<1, 32>>>(rand(), a, N, mean, std);
}

template <>
void init_normal<double>(double *a, int N, double mean, double std) {
     normal_kerneld<<<1, 32>>>(rand(), a, N, mean, std);
}

template <>
void add_normal<float>(float *a, int N, float mean, float std) {
     add_normal_kernel<<<1, 32>>>(rand(), a, N, mean, std);
}

template <>
void add_normal<double>(double *a, int N, double mean, double std) {
     add_normal_kerneld<<<1, 32>>>(rand(), a, N, mean, std);
}

__global__ void rand_init_kernel(int seed, curandStatePhilox4_32_10_t *states, int n) {
  int x(threadIdx.x + blockDim.x * blockIdx.x);

  if (x < n)
    curand_init(seed, x, 0, &states[x]);
}

__global__ void rand_zero_kernel(float *data, int n, float p, curandStatePhilox4_32_10_t *states) {
  int x(threadIdx.x + blockDim.x * blockIdx.x);

  curandStatePhilox4_32_10_t &state(states[x]);

  x *= 4;
  float4 vals = curand_uniform4(&state);
  for (int i(0); i < 4; ++i, ++x) {
    if (x >= n) return;
    if (reinterpret_cast<float*>(&vals)[i] < p)
      data[x] = 0;
  }
}

void rand_zero(float *data, int n, float p) {
  // assert(n > 1);
  static int n_rand_states = 0;
  static curandStatePhilox4_32_10_t* rand_states = 0;

  int const BLOCKSIZE(1024);
  int n_threads = (n + 4 - 1) / 4;

  dim3 dimBlock( BLOCKSIZE );
  dim3 dimGrid( (n_threads + BLOCKSIZE - 1) / BLOCKSIZE );

  if (n_threads > n_rand_states) {
	  if (rand_states) cudaFree(rand_states);
	  handle_error(cudaMalloc(&rand_states, sizeof(curandStatePhilox4_32_10_t) * n_threads));

    rand_init_kernel<<<dimGrid, dimBlock>>>(rand(), rand_states, n_threads);
    n_rand_states = n_threads;
  }


  rand_zero_kernel<<<dimGrid, dimBlock>>>(data, n, p, rand_states);
  handle_error( cudaGetLastError() );
  handle_error( cudaDeviceSynchronize());
}

__global__ void shift_kernel(float const *in, float *out, int X, int Y, int C, int dx, int dy, float const beta) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(x / X);
  x = x % X;

	int x_to(x + dx);
	int y_to(y + dy);
  // int x_to(x);
  // int y_to(y);
  // y = 0;
  // y_to = 0;


	if (x >= X || y >= Y || x_to >= X || y_to >= Y || x_to < 0 || y_to < 0)
		return;
  if (beta>0)
    add_c(in + get_index(X, Y, 1, C, x, y, 0), out + get_index(X, Y, 1, C, x_to, y_to, 0), X * Y, X * Y, C);
  else
    copy_c(in + get_index(X, Y, 1, C, x, y, 0), out + get_index(X, Y, 1, C, x_to, y_to, 0), X * Y, X * Y, C);

}

__global__ void unshift_kernel(float const *in, float *out, int X, int Y, int C, int dx, int dy, float const beta) {
  int x(threadIdx.x + blockDim.x * blockIdx.x);
  int y(x / X);
  x = x % X;

  int x_to(x + dx);
  int y_to(y + dy);

  if (x >= X || y >= Y || x_to >= X || y_to >= Y || x_to < 0 || y_to < 0)
    return;

  if (beta>0)
  	add_c(in + get_index(X, Y, 1, C, x_to, y_to, 0), out + get_index(X, Y, 1, C, x, y, 0), X * Y, X * Y, C);
  else
    copy_c(in + get_index(X, Y, 1, C, x_to, y_to, 0), out + get_index(X, Y, 1, C, x, y, 0), X * Y, X * Y, C);
}

void shift(float const *in, float *out, int X, int Y, int C, int dx, int dy, float const beta) {
	int s = X * Y;// * C;
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (s  + BLOCKSIZE - 1) / BLOCKSIZE);

	shift_kernel<<<dimGrid, dimBlock>>>(in, out, X, Y, C, dx, dy, beta);
}

void unshift(float const *in, float *out, int X, int Y, int C, int dx, int dy, float const beta) {
	int s = X * Y;// * C;
	int const BLOCKSIZE(1024);

	int dimBlock( BLOCKSIZE );
	int dimGrid( (s  + BLOCKSIZE - 1) / BLOCKSIZE);

	unshift_kernel<<<dimGrid, dimBlock>>>(in, out, X, Y, C, dx, dy, beta);
}
