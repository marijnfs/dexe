#ifndef __LOSS_H__
#define __LOSS_H__

#include "tensor.h"
#include "util.h"
#include <vector>

template <typename F>
struct Loss {
	Loss(int n, int c);
	
	virtual void calculate_loss(Tensor<F> &in, std::vector<int> answers, Tensor<F> &err) = 0;
	virtual void calculate_loss(Tensor<F> &in, int answer, Tensor<F> &err);
	virtual void calculate_loss(Tensor<F> &in, Tensor<F> &target, Tensor<F> &err) = 0;
	virtual void calculate_average_loss(Tensor<F> &in, Tensor<F> &err) { throw StringException("not implemented"); }

	virtual F loss();
	virtual int n_correct();

	int n, c;

	F last_loss;
	int last_correct;
};

template <typename F>
struct SoftmaxLoss : public Loss<F> {
	SoftmaxLoss(int n, int c);

	void calculate_loss(Tensor<F> &in, std::vector<int> answers, Tensor<F> &err);
    void calculate_average_loss(Tensor<F> &in, Tensor<F> &err);
	void calculate_loss(Tensor<F> &in, Tensor<F> &target, Tensor<F> &err);
};

template <typename F>
struct SquaredLoss : public Loss<F> {
	SquaredLoss(int n, int c);
	
	void calculate_loss(Tensor<F> &in, std::vector<int> answers, Tensor<F> &err);
};

#endif
