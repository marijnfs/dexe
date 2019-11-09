#pragma once

#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>

size_t const WORKSPACE_SIZE = size_t(128) * 1024 * 1024;

struct Handler {
  Handler();
  ~Handler();
  void init_handler();

  static cudnnHandle_t &cudnn();
  static curandGenerator_t &curand();
  static cublasHandle_t &cublas();
  static char *workspace();
  static size_t workspace_size() { return WORKSPACE_SIZE; }
  static void set_device(int n);

  static void s_init();
  static void deinit();

  cudnnHandle_t h_cudnn;
  curandGenerator_t h_curand;
  cublasHandle_t h_cublas;
     
  char *s_workspace;
    
    
  static Handler *s_handler;
};
