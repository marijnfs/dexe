#include "operations.h"
#include "handler.h"
#include "kernels.h"
#include <cublas_v2.h>
#include <cassert>

using namespace std;


template <typename F>
ConvolutionOperation<F>::ConvolutionOperation(int in_map_, int out_map_, int kw_, int kh_, bool keep_, size_t workspace_limit_):
	in_map(in_map_),
	out_map(out_map_),
	kw(kw_),
	kh(kh_),
	filter_bank(in_map_, out_map_, kw_, kh_),
	filter_bank_grad(in_map_, out_map_, kw_, kh_),
	bias(1, out_map_, 1, 1),
	bias_grad(1, out_map_, 1, 1),
	algo(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM), //default algorithm
	workspace(0),
	workspace_size(workspace_limit_),
	keep(keep_),
	rollout(false)
{
	int pad_h(0), pad_w(0), stride_w(1), stride_h(1), upscalex(1), upscaley(1);
	if (keep) {
		pad_w = kw / 2;
		pad_h = kh / 2;
	}
	// cout << "weight buffer: " << filter_bank.n_weights() << endl;
	// cout << "bias buffer: " << bias.size() << endl;
	//todo: calculate padding
	handle_error( cudnnCreateConvolutionDescriptor(&conv));
	handle_error( cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride_h, stride_w, upscalex, upscaley, CUDNN_CROSS_CORRELATION));
	//handle_error( cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride_h, stride_w, upscalex, upscaley, CUDNN_CONVOLUTION));
}


template <typename F>
ConvolutionOperation<F>::ConvolutionOperation(string dummy, int in_map_, int out_map_, int kw_, int kh_, int z, bool keep_, size_t workspace_limit_):
	in_map(in_map_),
	out_map(out_map_),
	kw(kw_),
	kh(kh_),
	filter_bank(in_map_, out_map_, kw_, kh_, z),
	filter_bank_grad(in_map_, out_map_, kw_, kh_, z),
	bias(1, out_map_, 1, 1),
	bias_grad(1, out_map_, 1, 1),
	algo(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM), //default algorithm
	algo_bwd_filter(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1),
	algo_bwd(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1),
	workspace(0),
	workspace_size(workspace_limit_),
	workspace_bwd(0),
	workspace_size_bwd(workspace_limit_),
	workspace_bwd_filter(0),
	workspace_size_bwd_filter(workspace_limit_),
	keep(keep_),
	rollout(true)
{
	int pad_h(0), pad_w(0), stride_w(1), stride_h(1), upscalex(1), upscaley(1);
	if (keep) {
		pad_w = kw / 2;
		pad_h = kh / 2;
	}
	// cout << "weight buffer: " << filter_bank.n_weights() << endl;
	// cout << "bias buffer: " << bias.size() << endl;
	//todo: calculate padding
	handle_error( cudnnCreateConvolutionDescriptor(&conv));
	handle_error( cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride_h, stride_w, upscalex, upscaley, CUDNN_CROSS_CORRELATION));
	//handle_error( cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride_h, stride_w, upscalex, upscaley, CUDNN_CONVOLUTION));
}

template <typename F>
void ConvolutionOperation<F>::update(F lr) {
	// cout << filter_bank_grad.to_vector() << endl;

	// cout << filter_bank_grad.to_vector() << " " << bias_grad.to_vector() << endl;
	// cout << filter_bank.to_vector() << " " << bias.to_vector() << endl;
	add_cuda<F>(filter_bank_grad.ptr(), filter_bank.ptr(), filter_bank.n_weights(), lr);
	add_cuda<F>(bias_grad.ptr(), bias.ptr(), bias.size(), lr * .1);
}

template <typename F>
void ConvolutionOperation<F>::l2(F l) {
	add_cuda<F>(filter_bank.ptr(), filter_bank_grad.ptr(), filter_bank.n_weights(), -l);
}

template <typename F>
void ConvolutionOperation<F>::init_normal(F mean, F std) {
	filter_bank.init_normal(mean, std);
	//bias.init_normal(mean, std);
}

template <typename F>
void ConvolutionOperation<F>::init_uniform(F var) {
	filter_bank.init_uniform(var);
//bias.init_uniform(var);
}

template <typename F>
vector<F> ConvolutionOperation<F>::to_vector() {
	vector<F> filter_values = filter_bank.to_vector();
	vector<F> bias_values = bias.to_vector();
	copy(bias_values.begin(), bias_values.end(), back_inserter(filter_values));
	return filter_values;
}

template <typename F>
void ConvolutionOperation<F>::from_vector(vector<F> &v) {
	assert(v.size() == filter_bank.n_weights() + bias.size());
	vector<F> filter_bank_weights(v.begin(), v.begin() + filter_bank.n_weights());
	filter_bank.from_vector(filter_bank_weights);

	vector<F> bias_weights(v.begin() + filter_bank.n_weights(), v.begin() + filter_bank.n_weights() + bias.size());
	bias.from_vector(bias_weights);
}

template <typename F>
int ConvolutionOperation<F>::size() {
	return filter_bank.n_weights() + bias.size();
}


template <typename F>
vector<F> ConvolutionOperation<F>::grad_to_vector() {
	vector<F> grad = filter_bank_grad.to_vector();
	vector<F> bias_grad_vec = bias_grad.to_vector();
	copy(bias_grad_vec.begin(), bias_grad_vec.end(), back_inserter(grad));
	return grad;
}

template <typename F>
void ConvolutionOperation<F>::forward(Tensor<F> &input, Tensor<F> &output, F beta) {
	F alpha(1.0);

	F alpha_bias(1), beta_bias(1);

	handle_error( cudnnConvolutionForward(Handler::cudnn(), &alpha, input.td, input.data, filter_bank.fd, filter_bank.weights, conv, algo, workspace, workspace_size, &beta, output.td, output.data));
	// handle_error( cudnnAddTensor(Handler::cudnn(), CUDNN_ADD_FEATURE_MAP, &alpha_bias, bias.td, bias.data, &beta_bias, output.td, output.data));
	handle_error( cudnnAddTensor(Handler::cudnn(), &alpha_bias, bias.td, bias.data, &beta_bias, output.td, output.data));
}


template <typename F>
void ConvolutionOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &output_grad, Tensor<F> &input_grad, F beta) {
	F alpha(1.0);
	handle_error( cudnnConvolutionBackwardData(Handler::cudnn(), &alpha, filter_bank.fd, filter_bank.weights, output_grad.td, output_grad.data, conv, algo_bwd, workspace_bwd, workspace_size_bwd, &beta, input_grad.td, input_grad.data) );
}

template <typename F>
void ConvolutionOperation<F>::backward_weights(Tensor<F> &input, Tensor<F> &output_grad, F beta) {
	F alpha_bias(1.0), beta_bias(beta);
	handle_error( cudnnConvolutionBackwardBias(Handler::cudnn(), &alpha_bias, output_grad.td, output_grad.data, &beta_bias, bias_grad.td, bias_grad.data) );

	F alpha(1.0);
	handle_error( cudnnConvolutionBackwardFilter(Handler::cudnn(), &alpha, input.td, input.data, output_grad.td, output_grad.data, conv, algo_bwd_filter, workspace_bwd_filter, workspace_size_bwd_filter, &beta, filter_bank_grad.fd, filter_bank_grad.weights) );
}

template <typename F>
void ConvolutionOperation<F>::zero_grad() {
	filter_bank_grad.zero();
	bias_grad.zero();
}

template <typename F>
TensorShape ConvolutionOperation<F>::output_shape(TensorShape in) {
	int x_even((filter_bank.kw + 1) % 2), y_even((filter_bank.kh + 1) % 2);
	return TensorShape{in.n, filter_bank.out_map, in.w + x_even, in.h + y_even};
}

template <typename F>
void ConvolutionOperation<F>::forward_dry_run(Tensor<F> &in, Tensor<F> &out) { // allocates workspace

	// handle_error( cudnnGetConvolutionForwardAlgorithm(Handler::cudnn(), in.td, filter_bank.fd, conv, out.td, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, workspace_size, &algo) );

	// handle_error( cudnnGetConvolutionBackwardDataAlgorithm( Handler::cudnn(), filter_bank.fd, out.td, conv, in.td,CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, workspace_size, &algo_bwd) );

	// handle_error( cudnnGetConvolutionBackwardFilterAlgorithm( Handler::cudnn(),in.td, out.td, conv, filter_bank.fd, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, workspace_size, &algo_bwd_filter) );
	handle_error( cudnnGetConvolutionForwardAlgorithm(Handler::cudnn(), in.td, filter_bank.fd, conv, out.td, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, workspace_size, &algo) );

	handle_error( cudnnGetConvolutionBackwardDataAlgorithm( Handler::cudnn(), filter_bank.fd, out.td, conv, in.td,CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, workspace_size, &algo_bwd) );

	handle_error( cudnnGetConvolutionBackwardFilterAlgorithm( Handler::cudnn(),in.td, out.td, conv, filter_bank.fd, CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, workspace_size, &algo_bwd_filter) );

	//algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
	//algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	// algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
	handle_error( cudnnGetConvolutionForwardWorkspaceSize(Handler::cudnn(), in.td, filter_bank.fd, conv, out.td, algo, &workspace_size) );
	handle_error( cudnnGetConvolutionBackwardDataWorkspaceSize(Handler::cudnn(), filter_bank_grad.fd, out.td, conv, in.td, algo_bwd, &workspace_size_bwd) );
	handle_error( cudnnGetConvolutionBackwardFilterWorkspaceSize(Handler::cudnn(), in.td, out.td, conv, filter_bank_grad.fd, algo_bwd_filter, &workspace_size_bwd_filter) );

	// if (workspace_size)
	cout << "workspace size: " << workspace_size << endl;
	cout << "workspace backward size: " << workspace_size_bwd << endl;

	if (workspace_size)
		handle_error( cudaMalloc( (void**)&workspace, workspace_size) );
	if (workspace_size_bwd)
		handle_error( cudaMalloc( (void**)&workspace_bwd, workspace_size_bwd) );
	if (workspace_size_bwd_filter)
		handle_error( cudaMalloc( (void**)&workspace_bwd_filter, workspace_size_bwd_filter) );
}


template <typename F>
void ConvolutionOperation<F>::scale_grad(F val) {
  scale_cuda(filter_bank_grad.ptr(), filter_bank_grad.n_weights(), val);
  scale_cuda(bias_grad.ptr(), bias_grad.size(), val);
  throw "";
}


template <typename F>
void ConvolutionOperation<F>::register_params(std::vector<CudaPtr<F> > &params, std::vector<CudaPtr<F>> &fast_params, std::vector<CudaPtr<F> > &grads, std::vector<CudaPtr<F> > &fast_grads) {
  //cout << "registering " << (rollout?"rollout":"no rollout") << endl;
	if (!rollout) {
		params.push_back(CudaPtr<F>{&filter_bank.weights, filter_bank.n_weights()});
		grads.push_back(CudaPtr<F>{&filter_bank_grad.weights, filter_bank_grad.n_weights()});
	} else {
		cout << "adding to fastweights " << filter_bank_grad.weights << " " << filter_bank_grad.n_weights() << " n=" << fast_grads.size() << endl;
		// params.push_back(CudaPtr<F>{&filter_bank.weights, filter_bank.n_weights()}); //HACK
		// grads.push_back(CudaPtr<F>{&filter_bank_grad.weights, filter_bank_grad.n_weights()});
		fast_params.push_back(CudaPtr<F>{&filter_bank.weights, filter_bank.n_weights()});
		fast_grads.push_back(CudaPtr<F>{&filter_bank_grad.weights, filter_bank_grad.n_weights()});
	}
	params.push_back(CudaPtr<F>{&bias.data, bias.size()});
	grads.push_back(CudaPtr<F>{&bias_grad.data, bias_grad.size()});
}

template <typename F>
void ConvolutionOperation<F>::share(ConvolutionOperation<F> &other){
	cudaFree(other.filter_bank.weights);
	cudaFree(other.bias.data);
	cudaFree(other.filter_bank_grad.weights);
	cudaFree(other.bias_grad.data);

	other.filter_bank.weights = filter_bank.weights;
	other.bias.data = bias.data;
	other.filter_bank_grad.weights = filter_bank_grad.weights;
	other.bias_grad.data = bias_grad.data;

}

///Timed Operations

template <typename F>
void ConvolutionOperation<F>::forward_timed(Tensor<F> &input, Tensor<F> &output, int t, F beta) {
	F alpha(1.0);

	F alpha_bias(1), beta_bias(1);

	handle_error( cudnnConvolutionForward(Handler::cudnn(), &alpha, input.td, input.data, filter_bank.fd, filter_bank.ptr(t), conv, algo, workspace, workspace_size, &beta, output.td, output.data));
	// handle_error( cudnnAddTensor(Handler::cudnn(), CUDNN_ADD_FEATURE_MAP, &alpha_bias, bias.td, bias.data, &beta_bias, output.td, output.data));
	handle_error( cudnnAddTensor(Handler::cudnn(), &alpha_bias, bias.td, bias.data, &beta_bias, output.td, output.data));
}


template <typename F>
void ConvolutionOperation<F>::backward_weights_timed(Tensor<F> &input, Tensor<F> &output_grad, int t, F beta) {
	F alpha_bias(1.0), beta_bias(beta);
	handle_error( cudnnConvolutionBackwardBias(Handler::cudnn(), &alpha_bias, output_grad.td, output_grad.data, &beta_bias, bias_grad.td, bias_grad.data) );

	F alpha(1.0);
	handle_error( cudnnConvolutionBackwardFilter(Handler::cudnn(), &alpha, input.td, input.data, output_grad.td, output_grad.data, conv, algo_bwd_filter, workspace_bwd_filter, workspace_size_bwd_filter, &beta, filter_bank_grad.fd, filter_bank_grad.ptr(t)) );

}

template <typename F>
void ConvolutionOperation<F>::backward_timed(Tensor<F> &in, Tensor<F> &out, Tensor<F> &output_grad, Tensor<F> &input_grad, int t, F beta) {
	F alpha(1.0);
	handle_error( cudnnConvolutionBackwardData(Handler::cudnn(), &alpha, filter_bank.fd, filter_bank.ptr(t), output_grad.td, output_grad.data, conv, algo_bwd, workspace_bwd, workspace_size_bwd, &beta, input_grad.td, input_grad.data) );
}


template <typename F>
ConvolutionOperation<F>::~ConvolutionOperation() {
	cudnnDestroyConvolutionDescriptor(conv);

    if (workspace)
		cudaFree(workspace);
}


/////////////Convolution Shift

template <typename F>
ConvolutionShiftOperation<F>::ConvolutionShiftOperation(int in_map, int out_map, int kw, int kh, int shift_x, int shift_y, bool keep_, size_t workspace_limit_):
	ConvolutionOperation<F>(in_map, out_map, kw, kh, keep_, workspace_limit_),
	dx(shift_x * (kw / 2)),
	dy(shift_y * (kh / 2))
{
	// int pad_h(0), pad_w(0), stride_w(1), stride_h(1), upscalex(1), upscaley(1);
	// if (this->keep) {
	// 	pad_w = kw / 2;
	// 	pad_h = kh / 2;
	// }
	// cout << "weight buffer: " << filter_bank.n_weights() << endl;
	// cout << "bias buffer: " << bias.size() << endl;
	//todo: calculate padding
	// handle_error( cudnnCreateConvolutionDescriptor(&(this->conv)));
	// handle_error( cudnnSetConvolution2dDescriptor(this->conv, pad_h, pad_w, stride_h, stride_w, upscalex, upscaley, CUDNN_CROSS_CORRELATION));
}

template <typename F>
ConvolutionShiftOperation<F>::~ConvolutionShiftOperation() {
	ConvolutionOperation<F>::~ConvolutionOperation();
}


template <typename F>
void ConvolutionShiftOperation<F>::forward(Tensor<F> &input, Tensor<F> &output, F beta) {
	F alpha(1.0);

	F alpha_bias(1), beta_bias(1);
	F slate_beta(0);
	handle_error( cudnnConvolutionForward(Handler::cudnn(), &alpha, input.td, input.data, this->filter_bank.fd, this->filter_bank.weights, this->conv, this->algo, this->workspace, this->workspace_size, &slate_beta, slate.td, slate.data));
	// handle_error( cudnnConvolutionForward(Handler::cudnn(), &alpha, input.td, input.data, this->filter_bank.fd, this->filter_bank.weights, this->conv, this->algo, this->workspace, this->workspace_size, &beta, output.td, output.data));//TESTING

	handle_error( cudnnAddTensor(Handler::cudnn(), &alpha_bias, this->bias.td, this->bias.data, &beta_bias, slate.td, slate.data));
	// handle_error( cudnnAddTensor(Handler::cudnn(), &alpha_bias, this->bias.td, this->bias.data, &beta_bias, output.td, output.data)); //TESTING
	// cout << "shapes: " << slate.shape() << " " << output.shape() << " " << this->bias.shape() << endl;
	// cout << "shifts: " << dx << " " << dy << endl;
	shift(slate.data, output.data, slate.w, slate.h, slate.c, dx, dy, beta);
	// shift(slate.data, output.data, slate.w, slate.h, slate.c, 0, 0, beta);
}

template <typename F>
void ConvolutionShiftOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &output_grad, Tensor<F> &input_grad, F beta) {
	F alpha(1.0);
	// slate_grad.zero();
	unshift(output_grad.data, slate_grad.data, slate_grad.w, slate_grad.h, slate_grad.c, dx, dy, 0);
	// handle_error( cudnnConvolutionBackwardData(Handler::cudnn(), &alpha, this->filter_bank.fd, this->filter_bank.weights, output_grad.td, output_grad.data, this->conv, this->algo_bwd, this->workspace_bwd, this->workspace_size_bwd, &beta, input_grad.td, input_grad.data) ); //TESTING
	handle_error( cudnnConvolutionBackwardData(Handler::cudnn(), &alpha, this->filter_bank.fd, this->filter_bank.weights, slate_grad.td, slate_grad.data, this->conv, this->algo_bwd, this->workspace_bwd, this->workspace_size_bwd, &beta, input_grad.td, input_grad.data) );

}

template <typename F>
void ConvolutionShiftOperation<F>::backward_weights(Tensor<F> &input, Tensor<F> &output_grad, F beta) {
	//Assuming backward_weights comes after backward, so no unshift
	//unshift(output_grad.data, slate_grad.data, slate_grad.w, slate_grad.h, slate_grad.c, dx, dy, 0);

	F alpha_bias(1.0), beta_bias(beta);
	handle_error( cudnnConvolutionBackwardBias(Handler::cudnn(), &alpha_bias, slate_grad.td, slate_grad.data, &beta_bias, this->bias_grad.td, this->bias_grad.data) );
	// handle_error( cudnnConvolutionBackwardBias(Handler::cudnn(), &alpha_bias, output_grad.td, output_grad.data, &beta_bias, this->bias_grad.td, this->bias_grad.data) );//TESTING

	F alpha(1.0);
	handle_error( cudnnConvolutionBackwardFilter(Handler::cudnn(), &alpha, input.td, input.data, slate_grad.td, slate_grad.data, this->conv, this->algo_bwd_filter, this->workspace_bwd_filter, this->workspace_size_bwd_filter, &beta, this->filter_bank_grad.fd, this->filter_bank_grad.weights) );
	// handle_error( cudnnConvolutionBackwardFilter(Handler::cudnn(), &alpha, input.td, input.data, output_grad.td, output_grad.data, this->conv, this->algo_bwd_filter, this->workspace_bwd_filter, this->workspace_size_bwd_filter, &beta, this->filter_bank_grad.fd, this->filter_bank_grad.weights) ); //TESTING
}


template <typename F>
void ConvolutionShiftOperation<F>::forward_dry_run(Tensor<F> &in, Tensor<F> &out) { // allocates workspace
	slate.reshape(out.shape());
	slate_grad.reshape(out.shape());

	ConvolutionOperation<F>::forward_dry_run(in, out);
	// handle_error( cudnnGetConvolutionForwardAlgorithm(Handler::cudnn(), in.td, this->filter_bank.fd, this->conv, out.td, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, this->workspace_size, &this->algo) );
	// //algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
	// //algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	// this->algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
	// handle_error( cudnnGetConvolutionForwardWorkspaceSize(Handler::cudnn(), in.td, this->filter_bank.fd, this->conv, out.td, this->algo, &(this->workspace_size)) );
	// if (this->workspace_size)
	// 	cout << "workspace size: " << this->workspace_size << endl;
	// if (this->workspace_size)
	// 	handle_error( cudaMalloc( (void**)&this->workspace, this->workspace_size) );
}

template <typename F>
void ConvolutionShiftOperation<F>::zero_grad() {
	this->filter_bank_grad.zero();
	this->bias_grad.zero();
	slate_grad.zero();
}

//////////////////////////////////////

template <typename F>
SquashOperation<F>::SquashOperation(TensorShape s, int c_) : c(c_), ConvolutionOperation<F>(s.c, c_, s.w, s.h, false) {

}

template <typename F>
TensorShape SquashOperation<F>::output_shape(TensorShape in) {
	return TensorShape{in.n, c, 1, 1};
}

template <typename F>
PoolingOperation<F>::PoolingOperation(int kw_, int kh_) : kw(kw_), kh(kh_) {
	handle_error( cudnnCreatePoolingDescriptor(&pool) );

	cudnnSetPooling2dDescriptor(pool, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, kw, kh, 0, 0, kw, kh);
}

template <typename F>
void PoolingOperation<F>::forward(Tensor<F> &in, Tensor<F> &out, F beta) {
	F alpha(1.0);
	handle_error( cudnnPoolingForward(Handler::cudnn(), pool, &alpha, in.td, in.data, &beta, out.td, out.data) );
}

template <typename F>
void PoolingOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta) {
	F alpha(1.0);
	handle_error( cudnnPoolingBackward(Handler::cudnn(), pool, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data) );
}

template <typename F>
TensorShape PoolingOperation<F>::output_shape(TensorShape in) {
	// cout << in.c << endl;
	return TensorShape{in.n, in.c, in.w / kw, in.h / kh};
}

template <typename F>
TanhOperation<F>::TanhOperation(F scale_) : scale(scale_) {
	cudnnCreateActivationDescriptor(&desc);
	cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0);
}

template <typename F>
void TanhOperation<F>::forward(Tensor<F> &in, Tensor<F> &out, F beta) {
  F alpha(1);
  handle_error( cudnnActivationForward(Handler::cudnn(), desc, &alpha, in.td, in.data, &beta, out.td, out.data));
  // tanh_forward<F>(in.data, out.data, out.size(), beta, scale);
}

template <typename F>
void TanhOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta) {
  F alpha(1);
  handle_error( cudnnActivationBackward(Handler::cudnn(), desc, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data));
  // tanh_deriv<F>(out_grad.data, out.data, in_grad.data, out.size(), beta, scale);
}

template <typename F>
TensorShape TanhOperation<F>::output_shape(TensorShape in) {
	return in;
}

template <typename F>
SigmoidOperation<F>::SigmoidOperation(F scale_) : scale(scale_) {
	cudnnCreateActivationDescriptor(&desc);
	cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0);
}

template <typename F>
void SigmoidOperation<F>::forward(Tensor<F> &in, Tensor<F> &out, F beta) {
  F alpha(1);
  sigm_forward<F>(in.data, out.data, out.size(), beta, scale);
  //cout << out.to_vector()[0];
  //handle_error( cudnnActivationForward(Handler::cudnn(), CUDNN_ACTIVATION_SIGMOID, &alpha, in.td, in.data, &beta, out.td, out.data));
}

template <typename F>
void SigmoidOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta) {
  //handle_error( cudnnActivationBackward(Handler::cudnn(), CUDNN_ACTIVATION_SIGMOID, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data));
  sigm_deriv<F>(out_grad.data, out.data, in_grad.data, out.size(), beta, scale);
}

template <typename F>
TensorShape SigmoidOperation<F>::output_shape(TensorShape in) {
	return in;
}

template <typename F>
ReluOperation<F>::ReluOperation() {
	cudnnCreateActivationDescriptor(&desc);
	cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 1.);
}

template <typename F>
void ReluOperation<F>::forward(Tensor<F> &in, Tensor<F> &out, F beta) {
  F alpha(1);
  handle_error( cudnnActivationForward(Handler::cudnn(), desc, &alpha, in.td, in.data, &beta, out.td, out.data));
}

template <typename F>
void ReluOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta) {
  F alpha(1);
  //handle_error( cudnnActivationBackward(Handler::cudnn(), CUDNN_ACTIVATION_RELU, &alpha, in.td, in.data, out_grad.td, out_grad.data, out.td, out.data, &beta, in_grad.td, in_grad.data));
  handle_error( cudnnActivationBackward(Handler::cudnn(), desc, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data));
}

template <typename F>
TensorShape ReluOperation<F>::output_shape(TensorShape in) {
	return in;
}

template <typename F>
void GateOperation<F>::forward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out, F beta) {
	gate(in, in2, out);
}

template <typename F>
void GateOperation<F>::backward(Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, Tensor<F> &in2_grad, F beta) {
	gate(out_grad, in2, in_grad);
	gate(out_grad, in, in2_grad);
}

template <typename F>
TensorShape GateOperation<F>::output_shape(TensorShape in) {
	return in;
}

template <typename F>
SoftmaxOperation<F>::SoftmaxOperation(bool matched_) : matched(matched_) {
}

template <typename F>
void SoftmaxOperation<F>::forward(Tensor<F> &in, Tensor<F> &out, F beta) {
	F alpha(1);
//	handle_error( cudnnSoftmaxForward(Handler::cudnn(), CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, in.td, in.data, &beta, out.td, out.data));
	handle_error( cudnnSoftmaxForward(Handler::cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, in.td, in.data, &beta, out.td, out.data));
}

template <typename F>
void SoftmaxOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta) {
	F alpha(1);
	//cout << out_grad.to_vector() << endl;
	//cout << in_grad.to_vector() << endl;
	//cout << out.to_vector() << endl;
	//cout << in.to_vector() << endl;

	if (matched) {//loss function matched
		in_grad.from_tensor(out_grad);
	}
	else
		handle_error( cudnnSoftmaxBackward(Handler::cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, out.td, out.data, out_grad.td, out_grad.data, &beta, in_grad.td, in_grad.data));
}

template <typename F>
TensorShape SoftmaxOperation<F>::output_shape(TensorShape in) {
	return in;
}

template struct ConvolutionOperation<float>;
template struct ConvolutionShiftOperation<float>;
template struct SquashOperation<float>;
template struct PoolingOperation<float>;
template struct TanhOperation<float>;
template struct SigmoidOperation<float>;
template struct ReluOperation<float>;
template struct SoftmaxOperation<float>;
template struct GateOperation<float>;

template struct ConvolutionOperation<double>;
template struct SquashOperation<double>;
template struct PoolingOperation<double>;
template struct TanhOperation<double>;
template struct SigmoidOperation<double>;
template struct ReluOperation<double>;
template struct SoftmaxOperation<double>;
template struct GateOperation<double>;
