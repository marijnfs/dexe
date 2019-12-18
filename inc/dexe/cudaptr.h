#pragma once

namespace dexe {

template <typename F>
struct CudaPtr {
	F **ptr;
	int n;

};

}
