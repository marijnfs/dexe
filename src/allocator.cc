#include "dexe/allocator.h"

namespace dexe {

void init_allocator() {
	if (allocator_stack.empty())
		allocator_stack.push(new DirectAllocator());
}

void push_allocator(Allocator *allocator) {
	// std::cout << "Dexe: Push Allocator" << std::endl;
	allocator_stack.push(allocator);
}

void pop_allocator() {
	// std::cout << "Dexe: Pop Allocator" << std::endl;
	allocator_stack.pop();
}

Allocator *get_allocator() {
	return allocator_stack.top();
}

}
