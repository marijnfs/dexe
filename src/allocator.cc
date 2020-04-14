#include "dexe/allocator.h"

namespace dexe {

void init_allocator() {
	if (allocator_stack.empty())
		allocator_stack.push(new DirectAllocator());
}

}
