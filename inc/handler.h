#ifndef __HANDLER_H__
#define __HANDLER_H__

#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>

struct Handler {
  Handler();
  ~Handler();
  void init_handler();

  static cudnnHandle_t &cudnn();
  static curandGenerator_t &curand();
  static cublasHandle_t &cublas();
  static void s_init();
  static void set_device(int n);

  cudnnHandle_t h_cudnn;
  curandGenerator_t h_curand;
  cublasHandle_t h_cublas;

  static Handler *s_handler;
};

#endif
