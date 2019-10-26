#include "normalise.h"

#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>

struct square
{
  square(float mean_) : mean(mean_){}
  __host__  __device__ float operator()(float x) const
  {
    x -= mean;
    return x * x;
  }
  
  float mean;
};


void normalise_fast(float *ptr, int n)
{
  //    const thrust::device_vector<float>& x)
  // with fusion

  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(ptr);
  float mean = thrust::reduce(dev_ptr, dev_ptr + n, 0, thrust::plus<float>()) / n;
    
  float std = sqrt( transform_reduce(dev_ptr, dev_ptr + n, square(mean), 0.0f, thrust::plus<float>())) / n;
  
  
  thrust::for_each(dev_ptr, dev_ptr + n, thrust::placeholders::_1 /= std);
  
}
