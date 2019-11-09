#pragma once

#include "network.h"
#include "tensor.h"
#include <vector>

struct Connet {
	std::vector<Tensor> tensors;

	std::vector<FilterBank> filters;
};
