#ifndef __HANDLER_H__
#define __HANDLER_H__

#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>

size_t const WORKSPACE_SIZE = size_t(2) * 1024 * 1024;

struct Handler {
  Handler();
  ~Handler();
  void init_handler();

  static cudnnHandle_t &cudnn();
  static curandGenerator_t &curand();
  static cublasHandle_t &cublas();
    static char *workspace();
  static void set_device(int n);

  static void s_init();
  static void deinit();

  cudnnHandle_t h_cudnn;
  curandGenerator_t h_curand;
  cublasHandle_t h_cublas;
     
  char *_workspace = nullptr;
    
    
  static Handler *s_handler;
};

#endif
