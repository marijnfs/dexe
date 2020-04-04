#pragma once

#include <iostream>
#include "dexe/tensor.h"

#ifdef USE_NIFTI
namespace dexe {
    std::unique_ptr<Tensor<float>> read_nifti(std::string &path);

}
#endif
