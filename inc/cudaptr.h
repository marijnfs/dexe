#ifndef __CUDAPTR_H__
#define __CUDAPTR_H__

template <typename F>
struct CudaPtr {
	F **ptr;
	int n;

};


#endif
