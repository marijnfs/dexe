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

	virtual ~Allocator() {
		std::cout << "= Destructor allocator" << std::endl;
		assert_slices_empty();
	};

	uint8_t *insert_slice(Slice slice) {
		std::cout << "insert slice: " << (size_t)slice.ptr << " " << slice.size << std::endl;
		auto match = std::lower_bound(slices.begin(), slices.end(), slice, [](Slice a, Slice b) -> bool {return a.ptr < b.ptr; });
		slices.insert(match, slice);
		return slice.ptr;
	}

	void erase_slice(uint8_t *ptr) {
		std::cout << "erase slice: " << (size_t)ptr << std::endl;

		Slice target{ptr, 0};

		auto match = std::lower_bound(slices.begin(), slices.end(), target, [](Slice a, Slice b) -> bool {return a.ptr < b.ptr; });
		if (match == slices.end() || match->ptr != ptr) {
			throw std::runtime_error("Free called on non allocated pointer");
		}
		slices.erase(match);
	}

	void assert_slices_empty() {
		if (slices.size() > 0) {
			std::ostringstream oss;
			oss << "ERROR: " << slices.size() << " slices left when calling allocator destructor" << std::endl;
			throw std::runtime_error(oss.str());
		}
	}

	std::vector<Slice> slices;
};


struct DirectAllocator : public Allocator {
	~DirectAllocator() {
		assert_slices_empty();
	}

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

struct DummyAllocator : public Allocator {
	uint8_t *allocate(size_t n_bytes) {
		return nullptr;
	}

	void free(uint8_t *ptr) {
    }

};

static std::stack<Allocator*> allocator_stack;

void init_allocator();
void push_allocator(Allocator *allocator);
void pop_allocator();
Allocator *get_allocator();

// Pretends to allocate, but actually just keeps track of what is allocated
// Used to calculate memory usage beforehand
struct VirtualAllocator : public Allocator {
	Slice master_slice;

	~VirtualAllocator() {
		assert_slices_empty();
	}

	//allocates n_bytes bytes
	uint8_t *allocate(size_t n_bytes) {
		std::cerr << "virtual allocate: " << n_bytes << std::endl;
		Slice suggested_slice{master_slice.ptr, n_bytes};

		if (slices.empty()) { //if there a no slices, simply add it
			master_slice.size = n_bytes;
			return insert_slice(suggested_slice);			
		}



		uint8_t *end_ptr = master_slice.ptr;
		for (auto slice : slices) {
			size_t available_size = slice.ptr - end_ptr;
			suggested_slice.ptr = end_ptr;
			suggested_slice.size = available_size;

			if (suggested_slice.size > n_bytes)
				break;
			end_ptr = slice.ptr + slice.size;
		}

		if (suggested_slice.size < n_bytes) { //doesn't fit in between
			suggested_slice.ptr = slices.back().ptr + slices.back().size;
			suggested_slice.size = n_bytes;
			master_slice.size = suggested_slice.ptr + suggested_slice.size - master_slice.ptr;
		} else {
			suggested_slice.size = n_bytes;
		}

		std::cout << "searched, n slices before: " << slices.size() << std::endl;
		return insert_slice(suggested_slice);
	}

	//frees previously allocated pointer ptr
	void free(uint8_t *ptr) {

		std::cerr << "virtual free: " << (size_t)ptr << std::endl;
		erase_slice(ptr);
	}

	size_t max_size() {
		return master_slice.size;
	}
};

// Allocates all memory before hand and maps is as needed onto that memory
struct MappedAllocator : public Allocator {
	Slice master_slice;

	MappedAllocator(size_t n_bytes) {
		handle_error(cudaMalloc((void **)&master_slice.ptr, n_bytes));
		master_slice.size = n_bytes;
	}

	~MappedAllocator() {
		handle_error(cudaFree((void*)master_slice.ptr));
		assert_slices_empty();
	}

	//allocates n_bytes bytes
	uint8_t *allocate(size_t n_bytes) {
		std::cerr << "virtual allocate: " << n_bytes << std::endl;
		Slice suggested_slice{master_slice.ptr, n_bytes};

		if (slices.empty()) { //if there a no slices, simply add it
			return insert_slice(suggested_slice);			
		}


		uint8_t *end_ptr = master_slice.ptr;
		for (auto slice : slices) {
			size_t available_size = slice.ptr - end_ptr;
			suggested_slice.ptr = end_ptr;
			suggested_slice.size = available_size;

			if (suggested_slice.size > n_bytes)
				break;
			end_ptr = slice.ptr + slice.size;
		}

		if (suggested_slice.size < n_bytes) { //doesn't fit in between
			suggested_slice.ptr = slices.back().ptr + slices.back().size;
			suggested_slice.size = n_bytes;
			if (master_slice.size < suggested_slice.ptr + suggested_slice.size - master_slice.ptr)
		 		throw std::runtime_error("Requested more memory than allocated in Page");
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