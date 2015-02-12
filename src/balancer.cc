#include "balancer.h"

using namespace std;

Balancer *Balancer::b = 0;

Balancer::Balancer() : cur(-1) {
}

void Balancer::start(int n) {
	Balancer::init();
	if (b->time.size() <= n) {
		b->time.resize(n + 1);
		b->factor.resize(n + 1, 1.0);
	}
	b->timer.start();
	b->cur = n;
}

void Balancer::stop(int n) {
	Balancer::init();
	b->update();
	b->cur = -1;
}

bool Balancer::ready(int n) {
	init();
	b->update();
	//cout << b->time << endl;
	return n == (min_element(b->time.begin(), b->time.end()) - b->time.begin());
}

void Balancer::advance(int n, float t) {
	init();
	for (size_t i(0); i < b->time.size(); ++i)
		if (i != n)
			b->time[i] += t * b->factor[i];
}

void Balancer::preference(int n, float f) {
	b->factor[n] = 1.0 / f;
}

void Balancer::update() {
	if (cur != -1)
		b->time[cur] += timer.since() * b->factor[cur];
	b->timer.start();
}

void Balancer::init(int n) {
	if (!b)
		b = new Balancer();
	if (n) {
		b->time.resize(n);
		b->factor.resize(n, 1.0);
	}
}
