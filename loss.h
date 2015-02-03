#ifndef __LOSS_H__
#define __LOSS_H__

#include "tensor.h"
#include <vector>

struct Loss {
	Loss(int n, int c);

	virtual void calculate_loss(Tensor &in, std::vector<int> answers, Tensor &err) = 0;
	virtual void calculate_loss(Tensor &in, int answer, Tensor &err);

	virtual float loss();
	virtual int n_correct();

	int n, c;

	float last_loss;
	int last_correct;
};

struct SoftmaxLoss : public Loss {
	SoftmaxLoss(int n, int c);

	void calculate_loss(Tensor &in, std::vector<int> answers, Tensor &err);
};

struct SquaredLoss : public Loss {
	SquaredLoss(int n, int c);

	void calculate_loss(Tensor &in, std::vector<int> answers, Tensor &err);
};

#endif
