#include "util.h"
#include "kernels.h"


template <>
void init_uniform(float *data, int n, float std) {
	// assert(n > 1);
	handle_error( curandGenerateUniform ( Handler::curand(), data, n) );
	range<float>(data, n, -std, std);
	//handle_error( curandGenerateNormal ( Handler::curand(), data, n, mean, std) );
	// if (n % 2)
	// 	handle_error( curandGenerateNormal ( Handler::curand(), data + (n / 2 * 2) - 1, 2, mean, std) );
}



template <>
void init_uniform(double *data, int n, double std) {
	// assert(n > 1);
	handle_error( curandGenerateUniformDouble ( Handler::curand(), data, n) );
	range<double>(data, n, -std, std);
	// handle_error( curandGenerateNormalDouble ( Handler::curand(), data, (n / 2 * 2), mean, std) );
	// if (n % 2)
	// 	handle_error( curandGenerateNormalDouble ( Handler::curand(), data + (n / 2 * 2) - 1, 2, mean, std) );
}
