#pragma once

#include <algorithm>
#include <vector>
#include <stack>
#include <cmath>
#include <stdint.h>


#include <cuda.h>
#include "util.h"

namespace dexe {

struct Slice {
	uint8_t *ptr = nullptr;
	size_t size = 0;
};

struct Allocator {
	virtual uint8_t *allocate(size_t n_bytes) = 0;
	virtual void free(uint8_t *ptr) = 0;

	virtual ~Allocator() = default;

	uint8_t *insert_slice(Slice slice) {
		auto match = std::lower_bound(slices.begin(), slices.end(), slice, [](Slice a, Slice b) -> bool {return a.ptr < b.ptr; });
		slices.insert(match, slice);
		return slice.ptr;
	}

	void erase_slice(uint8_t *ptr) {
		Slice target{ptr, 0};

		auto match = std::lower_bound(slices.begin(), slices.end(), target, [](Slice a, Slice b) -> bool {return a.ptr < b.ptr; });
		if (match->ptr != ptr) {
			throw std::runtime_error("Free called on non allocated pointer");
		}
		slices.erase(match);
	}


	std::vector<Slice> slices;
};


struct DirectAllocator : public Allocator {
	uint8_t *allocate(size_t n_bytes) {
		Slice slice;
        handle_error(cudaMalloc((void **)&slice.ptr, n_bytes));
        slice.size = n_bytes;
        return insert_slice(slice);
	}

	void free(uint8_t *ptr) {
		erase_slice(ptr);
        handle_error(cudaFree(ptr));
    }

};

static std::stack<Allocator*> allocator_stack;

void init_allocator();

// Pretends to allocate, but actually just keeps track of what is allocated
// Used to calculate memory usage beforehand
struct VirtualAllocator : public Allocator {
	Slice master_slice;

	//allocates n_bytes bytes
	uint8_t *allocate(size_t n_bytes) {
		Slice suggested_slice;

		uint8_t *ptr = master_slice.ptr;
		for (auto slice : slices) {
			size_t available_size = slice.ptr - ptr;
			if (available_size > suggested_slice.size) {
				suggested_slice.ptr = ptr;
				suggested_slice.size = available_size;
			}
			ptr = slice.ptr;
		}

		if (suggested_slice.size < n_bytes) { //doesn't fit in between
			suggested_slice.ptr = slices.back().ptr + slices.back().size;
			suggested_slice.size = n_bytes;
			master_slice.size = suggested_slice.ptr + suggested_slice.size - master_slice.ptr;
		} else {
			suggested_slice.size = n_bytes;
		}


		return insert_slice(suggested_slice);
	}

	//frees previously allocated pointer ptr
	void free(uint8_t *ptr) {
		erase_slice(ptr);
	}

	size_t max_size() {
		return master_slice.size;
	}
};

// Allocates all memory before hand and maps is as needed onto that memory
struct PagedAllocator : public Allocator {
	Slice master_slice;

	PagedAllocator(size_t n_bytes) {
		handle_error(cudaMalloc((void **)&master_slice.ptr, n_bytes));
		master_slice.size = n_bytes;
	}

	~PagedAllocator() {
		handle_error(cudaFree((void*)master_slice.ptr));
	}

	//allocates n_bytes bytes
	uint8_t *allocate(size_t n_bytes) {
		Slice suggested_slice;

		uint8_t *ptr = master_slice.ptr;
		for (auto slice : slices) {
			size_t available_size = slice.ptr - ptr;
			if (available_size > suggested_slice.size) {
				suggested_slice.ptr = ptr;
				suggested_slice.size = available_size;
			}
			ptr = slice.ptr;
		}

		if (suggested_slice.size < n_bytes) { //doesn't fit in between
			suggested_slice.ptr = slices.back().ptr + slices.back().size;
			suggested_slice.size = n_bytes;
			if (master_slice.size < suggested_slice.ptr + suggested_slice.size - master_slice.ptr) {
				throw std::runtime_error("Requested more memory than allocated in Page");
			}
		} else {
			suggested_slice.size = n_bytes;
		}

		return insert_slice(suggested_slice);
	}

	//frees previously allocated pointer ptr
	void free(uint8_t *ptr) {
		erase_slice(ptr);
	}
};

}