#pragma once

#include <iostream>
#include "dexe/tensor.h"

#ifdef USE_NIFTI
namespace dexe {
    std::unique_ptr<Tensor<float>> read_nifti(std::string path);
	void write_nifti(std::string path, Tensor<float> &data);

}
#endif
