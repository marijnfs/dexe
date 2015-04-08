#include "rand.h"

using namespace std;

Rand *Rand::s_rand = 0;

Rand::Rand() : engine(rd()) {
    // Choose a random mean between 1 and 6
    engine.seed(245);
}

Rand &Rand::inst() {
	if (!Rand::s_rand)
		Rand::s_rand = new Rand();
	return *Rand::s_rand;
}

int Rand::randn(int n) {
	uniform_int_distribution<int> uni(0, n - 1);
    return uni(Rand::inst().engine);
}
