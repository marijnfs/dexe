#pragma once

#include "util.h"
#include <vector>

namespace dexe {
struct Balancer {
	Balancer();

	static void start(int n);
	static void stop(int n);
	static bool ready(int n);
	static void advance(int n, float t);
	static void preference(int n, float f);
	static void init(int n = 0);
	void update();
	
	std::vector<float> time, factor;
	Timer timer;
	int cur;

	static Balancer *b;
};
}