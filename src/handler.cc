#include "handler.h"
#include "util.h"

Handler *Handler::s_handler = 0;

Handler::Handler():
  h_cudnn(0), h_cublas(0), h_curand(0)
{}

Handler::~Handler() {
  //todo: delete stuff
  cudaDeviceSynchronize();
}

void Handler::init_handler() {
  handle_error( cudnnCreate(&h_cudnn));

  //handle_error( curandCreateGenerator(&h_curand, CURAND_RNG_PSEUDO_XORWOW));
  handle_error( curandCreateGenerator(&h_curand,CURAND_RNG_PSEUDO_DEFAULT));
  handle_error( curandSetPseudoRandomGeneratorSeed(h_curand, 1234ULL));
  //handle_error( curandSetQuasiRandomGeneratorDimensions(h_curand, 1) );
  handle_error( cublasCreate(&h_cublas));
}

void Handler::set_device(int n) {
  handle_error( cudaSetDevice(n) );
}

void Handler::s_init() {
  s_handler = new Handler();
  s_handler->init_handler();
}

cudnnHandle_t &Handler::cudnn() {
  if (!s_handler)
    s_handler->s_init();
  return s_handler->h_cudnn;
}

curandGenerator_t &Handler::curand() {
  if (!s_handler)
    s_handler->s_init();
  return s_handler->h_curand;
}

cublasHandle_t &Handler::cublas() {
  if (!s_handler)
    s_handler->s_init();
  return s_handler->h_cublas;
}
