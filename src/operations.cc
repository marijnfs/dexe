#include "operations.h"
#include "handler.h"
#include "kernels.h"
#include <cublas_v2.h>
#include <cassert>

using namespace std;


template <typename F>
ConvolutionOperation<F>::ConvolutionOperation(vector<int> dimensions, vector<int> strides, bool keep_, size_t workspace_limit_):
	filter_bank(dimensions),
	filter_bank_grad(dimensions),
	algo(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM), //default algorithm
	workspace(0),
	workspace_size(workspace_limit_),
	keep(keep_)
{
	vector<int> bias_dims(dimensions.size(), 1); //number of filter dimensions is one more than output dim
	bias_dims[1] = filter_bank.out_c(); //dimensions[1] corresponds to output channels
	cout << "reshaping bias" << endl;
	bias.reshape(bias_dims);
	bias_grad.reshape(bias_dims);
	cout << "done reshaping bias" << endl;

	vector<int> kernel_dims(dimensions.begin() + 2, dimensions.end());
	vector<int> paddings(kernel_dims.size());
	vector<int> dilations(kernel_dims.size(), 1);

	if (keep) {
		cout << "pad: " << paddings << endl;
		for (int n(0); n < kernel_dims.size(); ++n)
			paddings[n] = kernel_dims[n] / 2;
		cout << "pad: " << paddings << endl;
	}

	handle_error( cudnnCreateConvolutionDescriptor(&conv));
	cout << "conv: " << kernel_dims << " " << paddings << " " << strides << " " << dilations << endl;
	cout << "bias dims: " << bias_dims << endl;
	if (sizeof(F) == sizeof(float))
		handle_error( cudnnSetConvolutionNdDescriptor(conv, kernel_dims.size(), paddings.data(), strides.data(), dilations.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
	else
		throw std::runtime_error("DOUBLE NONO");
	// handle_error( cudnnSetConvolutionNdDescriptor(conv, kernel_dims.size(), paddings.data(), strides.data(), dilations.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));
	//handle_error( cudnnSetConvolution2dDescriptor(conv, pad_h, pad_w, stride_h, stride_w, upscalex, upscaley, CUDNN_CONVOLUTION));
}


template <typename F>
void ConvolutionOperation<F>::forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) {
	forward(*in[0], *out[0]);
}

template <typename F>
bool ConvolutionOperation<F>::forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) {
	auto &in_tensor = *in[0];
	auto &out_tensor = *out[0];

	if (in_tensor.size() == 0) {
		cerr << "ConvolutionOperation: dry run failed, input size is zero" << endl;
		return false;
	}
	if (in_tensor.shape.c() != filter_bank.in_c()) {
		cerr << "ConvolutionOperation: input channels don't match filters" << endl;
		return false;
	}

	//reshape output tensor
	auto target_shape = in_tensor.shape;
	target_shape.set_c(filter_bank.out_c());
	out_tensor.reshape(target_shape);

	//prepare the workspaces
	forward_dry_run(in_tensor, out_tensor);

	return true;
}

template <typename F>
void ConvolutionOperation<F>::forward_dry_run(Tensor<F> &in, Tensor<F> &out) { // allocates workspace

	// handle_error( cudnnGetConvolutionForwardAlgorithm(Handler::cudnn(), in.td, filter_bank.fd, conv, out.td, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, workspace_size, &algo) );
	// handle_error( cudnnGetConvolutionBackwardDataAlgorithm( Handler::cudnn(), filter_bank.fd, out.td, conv, in.td,CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, workspace_size, &algo_bwd) );
	// handle_error( cudnnGetConvolutionBackwardFilterAlgorithm( Handler::cudnn(),in.td, out.td, conv, filter_bank.fd, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, workspace_size, &algo_bwd_filter) );
	cout << "dry run: " << workspace_size << endl;

	cout << in.shape << " " << filter_bank << " " << out.shape << endl;
	handle_error( cudnnGetConvolutionForwardAlgorithm(Handler::cudnn(), in.td, filter_bank.fd, conv, out.td, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, workspace_size, &algo) );
	// handle_error( cudnnGetConvolutionBackwardDataAlgorithm( Handler::cudnn(), filter_bank.fd, out.td, conv, in.td,CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, workspace_size, &algo_bwd) );
	// handle_error( cudnnGetConvolutionBackwardFilterAlgorithm( Handler::cudnn(),in.td, out.td, conv, filter_bank.fd, CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, workspace_size, &algo_bwd_filter) );

	// algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
	// algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	// algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
	handle_error( cudnnGetConvolutionForwardWorkspaceSize(Handler::cudnn(), in.td, filter_bank.fd, conv, out.td, algo, &workspace_size) );
	// handle_error( cudnnGetConvolutionBackwardDataWorkspaceSize(Handler::cudnn(), filter_bank_grad.fd, out.td, conv, in.td, algo_bwd, &workspace_size_bwd) );
	// handle_error( cudnnGetConvolutionBackwardFilterWorkspaceSize(Handler::cudnn(), in.td, out.td, conv, filter_bank_grad.fd, algo_bwd_filter, &workspace_size_bwd_filter) );

	// if (workspace_size)
	cout << "workspace size: " << workspace_size << endl;
	cout << "workspace backward size: " << workspace_size_bwd << endl;

	if (workspace_size) {
		if (workspace) {
			cudaFree(workspace);
			workspace = nullptr;
		}
		handle_error( cudaMalloc( (void**)&workspace, workspace_size) );
	}
	// if (workspace_size_bwd)
	// 	handle_error( cudaMalloc( (void**)&workspace_bwd, workspace_size_bwd) );
	// if (workspace_size_bwd_filter)
	// 	handle_error( cudaMalloc( (void**)&workspace_bwd_filter, workspace_size_bwd_filter) );
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
    cout << input.shape << " " << output.shape << " " << filter_bank << endl;
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
  F alpha_bias(1.0), beta_bias(beta / input.size());
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
	in.set_c(filter_bank.out_c());
	return in;
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
	params.push_back(CudaPtr<F>{&filter_bank.weights, filter_bank.n_weights()});
	grads.push_back(CudaPtr<F>{&filter_bank_grad.weights, filter_bank_grad.n_weights()});
	
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


template <typename F>
ConvolutionOperation<F>::~ConvolutionOperation() {
	cudnnDestroyConvolutionDescriptor(conv);

    if (workspace)
		cudaFree(workspace);
}


//////////////////////////////////////

template <typename F>
SquashOperation<F>::SquashOperation(TensorShape s, int c_) : c(c_), ConvolutionOperation<F>({s.c(), c_, s.d(), s.w(), s.h()}, {1, 1, 1, 1, 1}, false) {

}

template <typename F>
TensorShape SquashOperation<F>::output_shape(TensorShape in) {
	return TensorShape(in.n(), c, 1, 1);
}

template <typename F>
void SquashOperation<F>::init_normal(F mean, F std) {
  this->filter_bank.init_normal(mean, std / (this->filter_bank.kd() * this->filter_bank.kw() * this->filter_bank.kh()));
	//bias.init_normal(mean, std);
}

template <typename F>
void SquashOperation<F>::init_uniform(F var) {
  //this->filter_bank.init_uniform(var * (this->kw * this->kh));
  this->filter_bank.init_uniform(var);
  //bias.init_uniform(var);
}

////////////////// Unsquash

template <typename F>
UnsquashOperation<F>::UnsquashOperation(TensorShape s_) : s(s_) {

}

template <typename F>
TensorShape UnsquashOperation<F>::output_shape(TensorShape in) {
  return s;
}

template <typename F>
void UnsquashOperation<F>::forward(Tensor<F> &in, Tensor<F> &out, F beta) {
  out.from_tensor(in);
}

template <typename F>
void UnsquashOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta) {
  in_grad.from_tensor(out_grad);
}

////////////////// Merge

template <typename F>
MergeOperation<F>::MergeOperation() {
}

template <typename F>
TensorShape MergeOperation<F>::output_shape(TensorShape in) {
  return TensorShape{in.n(), in.c() / 4, in.w() * 2, in.h() * 2};
}

template <typename F>
void MergeOperation<F>::forward(Tensor<F> &in, Tensor<F> &out, F beta) {
  merge(in, out);
}

template <typename F>
void MergeOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta) {
  split(out_grad, in_grad);
}

////////////////// Split

template <typename F>
SplitOperation<F>::SplitOperation() {
  
}

template <typename F>
TensorShape SplitOperation<F>::output_shape(TensorShape in) {
  return TensorShape{in.n(), in.c() * 4, in.w() / 2, in.h() / 2};
}

template <typename F>
void SplitOperation<F>::forward(Tensor<F> &in, Tensor<F> &out, F beta) {
  split(in, out);
}

template <typename F>
void SplitOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta) {
  merge(out_grad, in_grad);
}



template <typename F>
PoolingOperation<F>::PoolingOperation(int kw_, int kh_, cudnnPoolingMode_t mode) : kw(kw_), kh(kh_) {
	handle_error( cudnnCreatePoolingDescriptor(&pool) );

	cudnnSetPooling2dDescriptor(pool, mode, CUDNN_NOT_PROPAGATE_NAN, kw, kh, 0, 0, kw, kh);
}

template <typename F>
void PoolingOperation<F>::forward(Tensor<F> &in, Tensor<F> &out, F beta) {
	F alpha(1.0);
	handle_error( cudnnPoolingForward(Handler::cudnn(), pool, &alpha, in.td, in.data, &beta, out.td, out.data) );
}

template <typename F>
void PoolingOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, F beta) {
	F alpha(1.0);
    //cout << in.shape() << " " << out.shape() << " " << in_grad.shape() << " " << out_grad.shape() << endl;
    //cout << in.data << " " << out.data << " " << out_grad.data << " " << in_grad.data << endl;
	handle_error( cudnnPoolingBackward(Handler::cudnn(), pool, &alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data, &beta, in_grad.td, in_grad.data) );
}

template <typename F>
TensorShape PoolingOperation<F>::output_shape(TensorShape in) {
	// cout << in.c << endl;
	return TensorShape{in.n(), in.c(), in.w() / kw, in.h() / kh};
}

template <typename F>
TanhOperation<F>::TanhOperation(F scale_) : scale(scale_) {
	cudnnCreateActivationDescriptor(&desc);
	cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0);
}

template <typename F>
void TanhOperation<F>::forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) {
	forward(*in[0], *out[0]);
}

template <typename F>
bool TanhOperation<F>::forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) {
	out[0]->reshape(in[0]->shape);
	return false;
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
AdditionOperation<F>::AdditionOperation() {
}

template <typename F>
void AdditionOperation<F>::forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) {
	forward(*in[0], *in[1], *out[0]);
}

template <typename F>
bool AdditionOperation<F>::forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) {
	auto &in_tensor0 = *in[0];
	auto &in_tensor1 = *in[1];
	auto &out_tensor = *out[0];

	if (in_tensor0.shape.n_elements() == 0 || in_tensor1.shape.n_elements() == 0) {
		cerr << "AdditionOperation: Input shape is empty" << endl;
		return false;
	}
	if (in_tensor0.shape != in_tensor1.shape) {
		cerr << "AdditionOperation: inputs don't match" << endl;
		return false;
	}

	out_tensor.reshape(in_tensor0.shape);

	return true;
}

template <typename F>
void AdditionOperation<F>::forward(Tensor<F> &in1, Tensor<F> &in2, Tensor<F> &out) {
  F alpha(1);

  add_cuda<F>(in1.ptr(), out.ptr(), in1.size(), float(1.0));
  add_cuda<F>(in2.ptr(), out.ptr(), in2.size(), 1.0);
}

template <typename F>
void AdditionOperation<F>::backward(Tensor<F> &out_grad, Tensor<F> &in_grad1, Tensor<F> &in_grad2) {
  F alpha(1);
  add_cuda<F>(out_grad.ptr(), in_grad1.ptr(), out_grad.size(), 1.0);
  add_cuda<F>(out_grad.ptr(), in_grad2.ptr(), out_grad.size(), 1.0);
}

template <typename F>
TensorShape AdditionOperation<F>::output_shape(TensorShape in) {
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
	cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.);
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
template struct SquashOperation<float>;
template struct UnsquashOperation<float>;
template struct MergeOperation<float>;
template struct SplitOperation<float>;
template struct AdditionOperation<float>;
template struct PoolingOperation<float>;
template struct TanhOperation<float>;
template struct SigmoidOperation<float>;
template struct ReluOperation<float>;
template struct SoftmaxOperation<float>;

// template struct ConvolutionOperation<double>;
// template struct SquashOperation<double>;
// template struct PoolingOperation<double>;
// template struct TanhOperation<double>;
// template struct SigmoidOperation<double>;
// template struct ReluOperation<double>;
// template struct SoftmaxOperation<double>;
