#pragma once

template <typename F>
struct CudaPtr {
	F **ptr;
	int n;

};
