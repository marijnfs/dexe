#pragma once

#include <random>

namespace dexe {

struct Rand {
	std::random_device rd;
	std::default_random_engine engine;

	Rand();

	static int randn(int n);

	static Rand &inst();
	static Rand *s_rand;
};

}
