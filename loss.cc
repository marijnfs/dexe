#include "loss.h"
#include <cmath>

using namespace std;

Loss::Loss(int n_, int c_) : n(n_), c(c_), last_loss(0), last_correct(0) {

}
	
void Loss::calculate_loss(Tensor<float> &in, int answer, Tensor<float> &err) {
    vector<int> answers(1);
	answers[0] = answer;
	calculate_loss(in, answers, err);
}

float Loss::loss() {
	return last_loss;
}

int Loss::n_correct() {
	return last_correct;
}

SoftmaxLoss::SoftmaxLoss(int n_, int c_) : Loss(n_, c_) {
}

void SoftmaxLoss::calculate_loss(Tensor<float> &in, vector<int> answers, Tensor<float> &err) {
    last_loss = 0;
	last_correct = 0;
	const float e(.00000001);
	vector<float> err_v(err.size());
	vector<float> prob = in.to_vector();

    for (size_t i(0); i < answers.size(); i++) {
        err_v[answers[i] + i * c] = 1.0;

		last_loss += -log(prob[answers[i]] + e);
		int max(0);
		float max_prob(0);
		for (size_t n(0); n < c; ++n)
			if (prob[n] > max_prob) {
				max_prob = prob[n];
				max = n;
			}
		if (max == answers[i]) ++last_correct;
	}

	err.from_vector(err_v);
	err -= in;
	//cout << "err: " << err.to_vector() << endl;
}

SquaredLoss::SquaredLoss(int n_, int c_) : Loss(n_, c_) {
}

void SquaredLoss::calculate_loss(Tensor<float> &in, vector<int> answers, Tensor<float> &err) {
	last_loss = 0;
	last_correct = 0;

	vector<float> err_v(err.size());
	vector<float> prob = in.to_vector();

	for (size_t i(0); i < answers.size(); i++) {
		err_v[answers[i] + i * c] = 1.0;
		for (size_t n(0); n < c; ++n)
			if (n == answers[i])
				last_loss += .5 * (prob[n] - 1.0) * (prob[n] - 1.0);
			else
				last_loss += .5 * (prob[n] - 0.0) * (prob[n] - 0.0);

		//cout << prob << endl;
		int max(0);
		float max_prob(0);
		for (size_t n(0); n < c; ++n)
			if (prob[n] > max_prob) {
				max_prob = prob[n];
				max = n;
			}
		if (max == answers[i]) ++last_correct;
	}

	err.from_vector(err_v);
	err -= in;
	//cout << "err: " << err.to_vector() << endl;
}

