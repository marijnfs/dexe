#ifndef __CONNET_H__
#define __CONNET_H__

#include "network.h"
#include "tensor.h"
#include <vector>

struct Connet {
	std::vector<Tensor> tensors;

	std::vector<FilterBank> filters;
};

#endif
