#pragma once

namespace dexe {

#include "dexe/network.h"
#include "dexe/tensor.h"
#include <vector>

struct Connet {
	std::vector<Tensor> tensors;

	std::vector<FilterBank> filters;
};

}
