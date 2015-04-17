#include "util.h"
#include "handler.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>

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

template <>
void init_normal<float>(float *a, int N, float mean, float std) {
     normal_kernel<<<1, 32>>>(rand(), a, N, mean, std);
}

template <>
void init_normal<double>(double *a, int N, double mean, double std) {
     normal_kerneld<<<1, 32>>>(rand(), a, N, mean, std);
}

__global__ void rand_zero_kernel(int seed, float *data, int n, float p) {
  int x(threadIdx.x + blockDim.x * blockIdx.x);
  if (x >= n) return;

  curandState state;
  curand_init(seed, x, 0, &state);

  if (curand_uniform(&state) < p)
    data[x] = 0;
}

void rand_zero(float *data, int n, float p) {
  // assert(n > 1);
  int const BLOCKSIZE(1024);

  dim3 dimBlock( BLOCKSIZE );
  dim3 dimGrid( (n + BLOCKSIZE - 1) / BLOCKSIZE );

  rand_zero_kernel<<<dimGrid, dimBlock>>>(rand(), data, n, p);
  handle_error( cudaGetLastError() );
  handle_error( cudaDeviceSynchronize());
}