#include "loss.h"
#include <cmath>

using namespace std;

template <typename F>
Loss<F>::Loss(int n_, int c_) : n(n_), c(c_), last_loss(0), last_correct(0) {

}

template <typename F>
void Loss<F>::calculate_loss(Tensor<F> &in, int answer, Tensor<F> &err) {
    vector<int> answers(1);
	answers[0] = answer;
	calculate_loss(in, answers, err);
}

template <typename F>
F Loss<F>::loss() {
	return this->last_loss;
}

template <typename F>
int Loss<F>::n_correct() {
	return last_correct;
}

template <typename F>
SoftmaxLoss<F>::SoftmaxLoss(int n_, int c_) : Loss<F>(n_, c_) {
}

template <typename F>
void SoftmaxLoss<F>::calculate_loss(Tensor<F> &in, vector<int> answers, Tensor<F> &err) {
    this->last_loss = 0;
    this->last_correct = 0;
	const F e(.00000001);
	vector<F> err_v(err.size());
	vector<F> prob = in.to_vector();

    for (size_t i(0); i < answers.size(); i++) {
        err_v[answers[i] + i * Loss<F>::c] = 1.0;

        Loss<F>::last_loss += -log(prob[answers[i] + i * Loss<F>::c] + e);
		int max(0);
		F max_prob(0);
        for (size_t n(0); n < Loss<F>::c; ++n)
			if (prob[n + i * Loss<F>::c] > max_prob) {
				max_prob = prob[n + i * Loss<F>::c];
				max = n;
			}
            if (max == answers[i]) ++Loss<F>::last_correct;
	}

	err.from_vector(err_v);
	err -= in;
	//cout << "err: " << err.to_vector() << endl;
}

template <typename F>
void SoftmaxLoss<F>::calculate_average_loss(Tensor<F> &in, Tensor<F> &err) {
    this->last_loss = 0;
    this->last_correct = 0;
	const F e(.00000001);
	vector<F> err_v(err.size());

	F guess = 1.0 / in.c;

	err.fill(guess);
	vector<F> prob = in.to_vector();

    for (size_t i(0); i < prob.size(); ++i)
		this->last_loss += -log(prob[i] + e);
	this->last_loss *= guess;

	err.from_vector(err_v);
	err -= in;
	//cout << "err: " << err.to_vector() << endl;
}

template <typename F>
void SoftmaxLoss<F>::calculate_loss(Tensor<F> &in, Tensor<F> &target, Tensor<F> &err) {
    this->last_loss = 0;
	//Loss<F>::last_loss = 0;
    this->last_correct = 0;

	//err.zero();
	err.from_tensor(target);
	err -= in;
	vector<F> errs = err.to_vector();
	//cout << "calculate loss bla";
	for (auto v : errs) {
		//cout << "vv:" << (v * v) << " ";
		this->last_loss += v * v;
	}
	//cout << "err: " << err.to_vector() << endl;
}

template <typename F>
SquaredLoss<F>::SquaredLoss(int n_, int c_) : Loss<F>(n_, c_) {
}

template <typename F>
void SquaredLoss<F>::calculate_loss(Tensor<F> &in, vector<int> answers, Tensor<F> &err) {
    this->last_loss = 0;
    this->last_correct = 0;

	vector<F> err_v(err.size());
	vector<F> prob = in.to_vector();

	for (size_t i(0); i < answers.size(); i++) {
       err_v[answers[i] + i * Loss<F>::c] = 1.0;
          for (size_t n(0); n < Loss<F>::c; ++n)
			if (n == answers[i])
				this->last_loss += .5 * (prob[n] - 1.0) * (prob[n] - 1.0);
			else
				this->last_loss += .5 * (prob[n] - 0.0) * (prob[n] - 0.0);

		//cout << prob << endl;
		int max(0);
		F max_prob(0);
        for (size_t n(0); n < Loss<F>::c; ++n)
			if (prob[n] > max_prob) {
				max_prob = prob[n];
				max = n;
			}
        if (max == answers[i]) ++Loss<F>::last_correct;
	}

	err.from_vector(err_v);
	err -= in;
	//cout << "err: " << err.to_vector() << endl;
}


template struct SquaredLoss<float>;
template struct SoftmaxLoss<float>;

template struct SquaredLoss<double>;
template struct SoftmaxLoss<double>;
