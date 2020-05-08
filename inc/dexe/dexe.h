#pragma once

#include "network.h"
#include "optimizer.h"
#include "handler.h"

namespace dexe {
	inline void init() {
		Handler::get_handler();
	}
}
