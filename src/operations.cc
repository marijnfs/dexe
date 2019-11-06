#include "operations.h"
#include "handler.h"
#include "kernels.h"
#include <cassert>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/set.hpp>


using namespace std;

template <typename F>
bool InputOperation<F>::forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) { 
	if (!reference)
		return true;
	if (reference->shape.c() != n_channels) {
		cerr << "input channels don't correspond data" << endl;
		return false;
	}
	out[0]->reshape(reference->shape);
	return true; 
}
     

template <typename F>
void InputOperation<F>::forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) {
	if (reference)
		out[0]->from_tensor(*reference);
}

template <typename F>
void InputOperation<F>::backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad) { 
}

template <typename F>
bool InputOperation<F>::backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad) { 
	//todo, check for needs_grad in input
	return true;
}

template <typename F>
InputOperation<F>::InputOperation(cereal::PortableBinaryInputArchive &ar) {
	ar(n_channels);
}

template <typename F>
void InputOperation<F>::save(cereal::PortableBinaryOutputArchive &ar) {
	ar(n_channels);
}


template <typename F>
ConvolutionOperation<F>::ConvolutionOperation(vector<int> dimensions_, vector<int> strides_, bool keep_, bool has_bias_, size_t workspace_limit_):
	//filter_bank(dimensions),
	//filter_bank_grad(dimensions),
	algo(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM), //default algorithm
	workspace(0),
	workspace_size(workspace_limit_),
	workspace_size_bwd(workspace_limit_),
	workspace_size_bwd_filter(workspace_limit_),
	keep(keep_),
    dimensions(dimensions_),
    strides(strides_),
    has_bias(has_bias_)
{
	init();
}

template <typename F>
void ConvolutionOperation<F>::init() {
	filter_bank.reshape(dimensions);
	
	if (has_bias) {
		vector<int> bias_dims(dimensions.size(), 1); //number of filter dimensions is one more than output dim
		bias_dims[1] = filter_bank.out_c(); //dimensions[1] corresponds to output channels
		cout << "reshaping bias to " << bias_dims << endl;
		bias.reshape(bias_dims);
		bias_grad.reshape(bias_dims);
		cout << "done reshaping bias" << endl;
	}
	
	vector<int> kernel_dims(dimensions.begin() + 2, dimensions.end());
	paddings = vector<int>(kernel_dims.size());
    dilations = vector<int>(kernel_dims.size(), 1);

	if (keep) {
		cout << "pad: " << paddings << endl;
		for (int n(0); n < kernel_dims.size(); ++n)
			paddings[n] = kernel_dims[n] / 2;
		cout << "pad: " << paddings << endl;
	}

	handle_error( cudnnCreateConvolutionDescriptor(&conv));
	cout << "conv: " << kernel_dims << " " << paddings << " " << strides << " " << dilations << endl;

	if (sizeof(F) == sizeof(float))
		handle_error( cudnnSetConvolutionNdDescriptor(conv, kernel_dims.size(), paddings.data(), strides.data(), dilations.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
	else
		handle_error( cudnnSetConvolutionNdDescriptor(conv, kernel_dims.size(), paddings.data(), strides.data(), dilations.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));	
}


template <typename F>
void ConvolutionOperation<F>::forward(vector<Tensor<F>*> &in, vector<Tensor<F>*> &out) {
	forward(*in[0], *out[0]);
}

template <typename F>
bool ConvolutionOperation<F>::backward_dry_run(vector<Tensor<F>*> &in, vector<Tensor<F>*> &out, vector<Tensor<F>*> &in_grad, vector<Tensor<F>*> &out_grad) {
    in_grad[0]->reshape(in[0]->shape);
    prepare_backward(*in[0], *out[0]);
    prepare_backward_weights(*in[0], *out[0]);
    return true;
}

template <typename F>
void ConvolutionOperation<F>::backward(vector<Tensor<F>*> &in, vector<Tensor<F>*> &out, vector<Tensor<F>*> &in_grad, vector<Tensor<F>*> &out_grad) {
    backward(*in[0], *out[0], *in_grad[0], *out_grad[0]);
    backward_weights(*in[0], *out_grad[0]);
}

template <typename F>
bool ConvolutionOperation<F>::check_fit(Tensor<F> &in_tensor, Tensor<F> &out_tensor) {
    // Check if input makes sense
	if (in_tensor.size() == 0) {
		cerr << "ConvolutionOperation: dry run failed, input size is zero" << endl;
		return false;
	}
	if (in_tensor.shape.c() != filter_bank.in_c()) {
		cerr << "ConvolutionOperation: input channels don't match filters" << endl;
		return false;
	}
    if (in_tensor.shape.n_dimensions() != out_tensor.shape.n_dimensions()) {
        cerr << "ConvolutionOperation: number of input dimensions don't match output dimensions" << endl;
		return false;
	}

    // check if strides divide    
    for (int n(0); n < paddings.size(); ++n) {
        if ((in_tensor.shape[n + 2] + 2 * paddings[n] - dimensions[n + 2]) % strides[n] != 0) {
            cerr << "Stride does not divide dimension" << endl;
            return false;
        }            
    }
    return true;
}

template <typename F>
bool ConvolutionOperation<F>::forward_dry_run(vector<Tensor<F>*> &in, vector<Tensor<F>*> &out) {
	auto &in_tensor = *in[0];
	auto &out_tensor = *out[0];
	if (!check_fit(in_tensor, out_tensor))
        return false;

	//prepare the workspaces
	prepare_forward(in_tensor, out_tensor);
	return true;
}

template <typename F>
void ConvolutionOperation<F>::prepare_forward(Tensor<F> &in, Tensor<F> &out) { // allocates workspace

	//reshape output tensor, calculated from paddings, strides and dimensions of filters
	auto target_shape = in.shape;
	target_shape.set_c(filter_bank.out_c());
    for (int n(0); n < paddings.size(); ++n) {
        target_shape[n + 2] = (in.shape[n + 2] + 2 * paddings[n] - dimensions[n + 2]) / strides[n] + 1;
    }
	out.reshape(target_shape);
	// algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
	// algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	// algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
	size_t new_workspace_size = Handler::workspace_size();
	//size_t new_workspace_size = workspace_size;

	handle_error( cudnnGetConvolutionForwardAlgorithm(Handler::cudnn(), in.td, filter_bank.fd, conv, out.td, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, new_workspace_size, &algo) );
	handle_error( cudnnGetConvolutionForwardWorkspaceSize(Handler::cudnn(), in.td, filter_bank.fd, conv, out.td, algo, &new_workspace_size) );

	workspace_size = new_workspace_size;
    workspace = Handler::workspace();
    /*
	if (workspace_size != new_workspace_size) {
		workspace_size = new_workspace_size;
		cout << "Allocating workspace of size: " << workspace_size << endl;
		if (workspace) {
			cudaFree(workspace);
			workspace = nullptr;
		}
		handle_error( cudaMalloc( (void**)&workspace, workspace_size) );
	}*/	
}

template <typename F>
void ConvolutionOperation<F>::prepare_backward_weights(Tensor<F> &in, Tensor<F> &out) { // allocates workspace
	//size_t new_workspace_size = workspace_size_bwd_filter;
	size_t new_workspace_size = Handler::workspace_size();

    handle_error( cudnnGetConvolutionBackwardFilterAlgorithm( Handler::cudnn(),in.td, out.td, conv, filter_bank.fd, CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, new_workspace_size, &algo_bwd_filter) );
    handle_error( cudnnGetConvolutionBackwardFilterWorkspaceSize(Handler::cudnn(), in.td, out.td, conv, filter_bank_grad.fd, algo_bwd_filter, &new_workspace_size) );

	workspace_size_bwd_filter = new_workspace_size;
    workspace_bwd_filter = Handler::workspace();
    /*
    if (workspace_size_bwd_filter != new_workspace_size) {
    	workspace_size_bwd_filter = new_workspace_size;
		if (workspace_bwd_filter) {
			cudaFree(workspace_bwd_filter);
			workspace_bwd_filter = nullptr;
		}
		handle_error( cudaMalloc( (void**)&workspace_bwd_filter, workspace_size_bwd_filter) );
    }*/
}

template <typename F>
void ConvolutionOperation<F>::prepare_backward(Tensor<F> &in, Tensor<F> &out) { // allocates workspace
	size_t new_workspace_size = Handler::workspace_size();
	//size_t new_workspace_size = workspace_size_bwd;

	handle_error( cudnnGetConvolutionBackwardDataAlgorithm(Handler::cudnn(), filter_bank.fd, out.td, conv, in.td, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, new_workspace_size, &algo_bwd) );
	handle_error( cudnnGetConvolutionBackwardDataWorkspaceSize(Handler::cudnn(), filter_bank.fd, out.td, conv, in.td, algo_bwd, &new_workspace_size) );

	workspace_size_bwd = new_workspace_size;
    workspace_bwd = Handler::workspace();
    /*
    if (workspace_size_bwd != new_workspace_size) {
    	workspace_size_bwd = new_workspace_size;
		if (workspace_bwd) {
			cudaFree(workspace_bwd);
			workspace_bwd = nullptr;
		}
		handle_error( cudaMalloc( (void**)&workspace_bwd, workspace_size_bwd) );
    }*/
}

template <typename F>
void ConvolutionOperation<F>::update(F lr) {
	// cout << filter_bank_grad.to_vector() << endl;

	// cout << filter_bank_grad.to_vector() << " " << bias_grad.to_vector() << endl;
	// cout << filter_bank.to_vector() << " " << bias.to_vector() << endl;
	add_cuda<F>(filter_bank_grad.ptr(), filter_bank.ptr(), filter_bank.n_weights(), lr);
	if (has_bias)
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
	if (has_bias) {
	 	vector<F> bias_values = bias.to_vector();
		copy(bias_values.begin(), bias_values.end(), back_inserter(filter_values));
	}
	return filter_values;
}

template <typename F>
void ConvolutionOperation<F>::from_vector(vector<F> &v) {
	assert(v.size() == filter_bank.n_weights() + bias.size());
	vector<F> filter_bank_weights(v.begin(), v.begin() + filter_bank.n_weights());
	filter_bank.from_vector(filter_bank_weights);

	if (has_bias) {
		vector<F> bias_weights(v.begin() + filter_bank.n_weights(), v.begin() + filter_bank.n_weights() + bias.size());
		bias.from_vector(bias_weights);
	}
}

template <typename F>
int ConvolutionOperation<F>::size() {
	return filter_bank.n_weights() + (has_bias ? bias.size() : 0);
}


template <typename F>
vector<F> ConvolutionOperation<F>::grad_to_vector() {
	vector<F> grad = filter_bank_grad.to_vector();
	if (has_bias) {
		vector<F> bias_grad_vec = bias_grad.to_vector();
		copy(bias_grad_vec.begin(), bias_grad_vec.end(), back_inserter(grad));
	}
	return grad;
}

template <typename F>
void ConvolutionOperation<F>::forward(Tensor<F> &input, Tensor<F> &output, F beta) {
	F alpha(1.0);

	F alpha_bias(1), beta_bias(1);
    // cout << input.shape << " " << output.shape << " " << filter_bank << endl;
    // cout << dimensions << " " << strides << " " << paddings << " " << dilations << endl; 
	// cout << filter_bank << " " << filter_bank.weights << endl;
	//cout << workspace << " " << workspace_size << endl;
	handle_error( cudnnConvolutionForward(Handler::cudnn(), &alpha, input.td, input.data, filter_bank.fd, filter_bank.weights, conv, algo, workspace, workspace_size, &beta, output.td, output.data));
	// handle_error( cudnnAddTensor(Handler::cudnn(), CUDNN_ADD_FEATURE_MAP, &alpha_bias, bias.td, bias.data, &beta_bias, output.td, output.data));
	// auto v = bias.to_vector();
	if (has_bias)
		handle_error( cudnnAddTensor(Handler::cudnn(), &alpha_bias, bias.td, bias.data, &beta_bias, output.td, output.data));
}


template <typename F>
void ConvolutionOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &input_grad, Tensor<F> &output_grad, F beta) {
	F alpha(1.0);
	// cout << ":in out filter: " << input_grad.shape << " " << output_grad.shape << " " << filter_bank << endl;
	// cout << strides << " " << dimensions << " " << paddings << endl;
	// algo_bwd = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
	handle_error( cudnnConvolutionBackwardData(Handler::cudnn(), &alpha, filter_bank.fd, filter_bank.weights, output_grad.td, output_grad.data, conv, algo_bwd, workspace_bwd, workspace_size_bwd, &beta, input_grad.td, input_grad.data) );
}

template <typename F>
void ConvolutionOperation<F>::backward_weights(Tensor<F> &input, Tensor<F> &output_grad, F beta) {
	if (has_bias) {
		F alpha_bias(1.0), beta_bias(beta / input.size());
		handle_error( cudnnConvolutionBackwardBias(Handler::cudnn(), &alpha_bias, output_grad.td, output_grad.data, &beta_bias, bias_grad.td, bias_grad.data) );
	}

	F alpha(1.0);
	handle_error( cudnnConvolutionBackwardFilter(Handler::cudnn(), &alpha, input.td, input.data, output_grad.td, output_grad.data, conv, algo_bwd_filter, workspace_bwd_filter, workspace_size_bwd_filter, &beta, filter_bank_grad.fd, filter_bank_grad.weights) );
}

template <typename F>
void ConvolutionOperation<F>::zero_grad() {
	filter_bank_grad.zero();
	if (has_bias)
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
  if (has_bias)
	  scale_cuda(bias_grad.ptr(), bias_grad.size(), val);
  throw "";
}


template <typename F>
void ConvolutionOperation<F>::register_params(vector<CudaPtr<F> > &params, vector<CudaPtr<F>> &fast_params, vector<CudaPtr<F> > &grads, vector<CudaPtr<F> > &fast_grads) {
  //cout << "registering " << (rollout?"rollout":"no rollout") << endl;
	params.push_back(CudaPtr<F>{&filter_bank.weights, filter_bank.n_weights()});
	grads.push_back(CudaPtr<F>{&filter_bank_grad.weights, filter_bank_grad.n_weights()});
	
	if (has_bias)
		params.push_back(CudaPtr<F>{&bias.data, bias.size()});
	grads.push_back(CudaPtr<F>{&bias_grad.data, bias_grad.size()});
}

template <typename F>
void ConvolutionOperation<F>::share(ConvolutionOperation<F> &other){
	cudaFree(other.filter_bank.weights);
	cudaFree(other.filter_bank_grad.weights);
	
	other.filter_bank.weights = filter_bank.weights;
	other.filter_bank_grad.weights = filter_bank_grad.weights;
	
	if (has_bias) {
		cudaFree(other.bias.data);
		cudaFree(other.bias_grad.data);

		other.bias.data = bias.data;
		other.bias_grad.data = bias_grad.data;
	}

}

template <typename F>
ConvolutionOperation<F>::~ConvolutionOperation() {
	cudnnDestroyConvolutionDescriptor(conv);

    if (workspace)
		cudaFree(workspace);
}


template <typename F>
void ConvolutionOperation<F>::save(cereal::PortableBinaryOutputArchive &ar) {
	ar(dimensions);
	ar(strides);
	ar(paddings);
	ar(dilations);
	ar(has_bias);
	ar(keep);

	ar(filter_bank.to_vector());
	if (has_bias)
		ar(bias.to_vector());
	
}

template <typename F>
ConvolutionOperation<F>::ConvolutionOperation(cereal::PortableBinaryInputArchive &ar) {
	ar(dimensions);
	ar(strides);
	ar(paddings);
	ar(dilations);
	ar(has_bias);
	ar(keep);
	
	init();
	
	vector<F> filter_bank_vec;
	ar(filter_bank_vec);
	filter_bank.from_vector(filter_bank_vec);

	if (has_bias) {
		vector<F> bias_vec;
		ar(bias_vec);
		bias.from_vector(bias_vec);
	}
}


///////////////////
template <typename F>
ConvolutionTransposeOperation<F>::ConvolutionTransposeOperation(std::vector<int> dimensions_, std::vector<int> strides_, bool keep_, size_t workspace_limit_) 
: ConvolutionOperation<F>(dimensions_, strides_, keep_, false, workspace_limit_) {
}

template <typename F>
void ConvolutionTransposeOperation<F>::forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out){
	Tensor<F> dummy;
	ConvolutionOperation<F>::backward(dummy, dummy, *out[0], *in[0]); //we use ConvolutionOperation in reverse to get the transpose
}

template <typename F>
bool ConvolutionTransposeOperation<F>::forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out){
    auto in_shape = in[0]->shape;
    //In the transpose we use cudnnConvolutions in reverse, so the output dimension is not the 1st dimension, input the 0th.
    if (in_shape.c() != this->dimensions[0]) {
        cerr << "channel dimension doesn't fit filter" << endl;
        return false;
    }

    auto output_shape = in_shape;
    output_shape.set_c(this->dimensions[1]);
    
    //Check and set the dimensions for every image dimension
    cout << "shape: " << in_shape << endl;
    for (int n(0); n < this->paddings.size(); ++n) {
        // in = (out - 1) * stride + dim - 2 * paddings
        auto intermediate = (in_shape[n + 2] - 1) * this->strides[n] + this->dimensions[n + 2];
        if (intermediate <= 2 * this->paddings[n]) {
            cerr << "paddings would cut off too much" << endl;
            return false;
        }
        output_shape[n + 2] = intermediate - 2 * this->paddings[n];
    }
    out[0]->reshape(output_shape);

    Tensor<F> dummy;
	ConvolutionOperation<F>::prepare_backward(*out[0], *in[0]);
    return true;
}

template <typename F>
void ConvolutionTransposeOperation<F>::backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad){
	ConvolutionOperation<F>::forward(*out_grad[0], *in_grad[0]); //we use ConvolutionOperation in reverse to get the transpose	
	ConvolutionOperation<F>::backward_weights(*out_grad[0], *in[0]); //we use ConvolutionOperation in reverse to get the transpose	
}

template <typename F>
bool ConvolutionTransposeOperation<F>::backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad){
	ConvolutionOperation<F>::prepare_forward(*out_grad[0], *in_grad[0]);
	ConvolutionOperation<F>::prepare_backward_weights(*out_grad[0], *in[0]); 
	return true;
}

template <typename F>
ConvolutionTransposeOperation<F>::ConvolutionTransposeOperation(cereal::PortableBinaryInputArchive &ar) : ConvolutionOperation<F>(ar) {
}

template <typename F>
void ConvolutionTransposeOperation<F>::save(cereal::PortableBinaryOutputArchive &ar) {
	ConvolutionOperation<F>::save(ar);
}

/////////////////////
template <typename F>
SquaredLossOperation<F>::SquaredLossOperation() {

}

template <typename F>
void SquaredLossOperation<F>::forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) {
	//todo: prealloc tmp or make dedicated kernel
	tmp.from_tensor(*in[0]);
	tmp.add(*in[1], -1.0);
	F loss = tmp.norm2() / tmp.shape.n_elements() / 2.0;
	out[0]->from_ptr(&loss);
}

template <typename F>
bool SquaredLossOperation<F>::forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) {
	if (in[0]->shape != in[1]->shape) {
		cerr << "input shapes don't match, " << in[0]->shape << " != " << in[1]->shape << endl;
		return false;
	}

	out[0]->reshape(TensorShape({1, 1, 1}));
	tmp.reshape(in[0]->shape);
	return true;
}

template <typename F>
bool SquaredLossOperation<F>::backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad) {
	in_grad[0]->reshape(in[0]->shape);
    return true;
}

template <typename F>
void SquaredLossOperation<F>::backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad) {
	//in[0] = prediction, in[1] = target
	in_grad[0]->from_tensor(*in[1]);
	in_grad[0]->add(*in[0], -1.0);
	in_grad[0]->scale(1.0 / in[0]->shape.n_elements());	
}



//////////////////////////////////////

template <typename F>
SquashOperation<F>::SquashOperation(TensorShape s, int c_) : c(c_), ConvolutionOperation<F>({s.c(), c_, s.d(), s.w(), s.h()}, {1, 1, 1, 1, 1}, false) {
    
}

template <typename F>
TensorShape SquashOperation<F>::output_shape(TensorShape in) {
  vector<int> out_dimensions(in.n_dimensions(), 1);
  out_dimensions[0] = in.n();
  out_dimensions[1] = c;
  return out_dimensions;
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
void UnsquashOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta) {
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
void MergeOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta) {
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
void SplitOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta) {
  merge(out_grad, in_grad);
}

/////////// LocalNormalisationOperation

template <typename F>
LocalNormalisationOperation<F>::LocalNormalisationOperation(int w_) :  w(w_) {
	handle_error( cudnnCreateLRNDescriptor( &lrn_desc ) );
	// handle_error( cudnnSetLRNDescriptor( lrn_desc, w, 1.0, 1.0e-4, 0.75) );
}

template <typename F>
TensorShape LocalNormalisationOperation<F>::output_shape(TensorShape input) {
	return input;
}

template <typename F>
void LocalNormalisationOperation<F>::forward(Tensor<F> &in, Tensor<F> &out, F beta) {
	F alpha(1.0);
 	handle_error( cudnnLRNCrossChannelForward(Handler::cudnn(), lrn_desc, CUDNN_LRN_CROSS_CHANNEL_DIM1, 
 				&alpha, in.td, in.data,
 				&beta, out.td, out.data) );
}

template <typename F>
void LocalNormalisationOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta) {
	F alpha(1.0);
	handle_error( cudnnLRNCrossChannelBackward(Handler::cudnn(), lrn_desc, CUDNN_LRN_CROSS_CHANNEL_DIM1, 
 				&alpha, out.td, out.data, out_grad.td, out_grad.data, in.td, in.data,
 				&beta, in_grad.td, in_grad.data) );
}
	
template <typename F>
LocalNormalisationOperation<F>::LocalNormalisationOperation(cereal::PortableBinaryInputArchive &ar) {
	ar(w);
	handle_error( cudnnCreateLRNDescriptor( &lrn_desc ) );	
}


template <typename F>
void LocalNormalisationOperation<F>::save(cereal::PortableBinaryOutputArchive &ar) {
	ar(w);
}

/////////// PoolingOperation

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
void PoolingOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta) {
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
void TanhOperation<F>::forward(vector<Tensor<F>*> &in, vector<Tensor<F>*> &out) {
	forward(*in[0], *out[0]);
}

template <typename F>
bool TanhOperation<F>::forward_dry_run(vector<Tensor<F>*> &in, vector<Tensor<F>*> &out) {
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
void TanhOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta) {
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
void AdditionOperation<F>::forward(vector<Tensor<F>*> &in, vector<Tensor<F>*> &out) {
	forward(*in[0], *in[1], *out[0]);
}

template <typename F>
bool AdditionOperation<F>::forward_dry_run(vector<Tensor<F>*> &in, vector<Tensor<F>*> &out) {
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
void AdditionOperation<F>::backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad) {
	backward(*out_grad[0], *in_grad[0], *in_grad[1]);
}

template <typename F>
bool AdditionOperation<F>::backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad) {
	in_grad[0]->reshape(in[0]->shape);
	in_grad[1]->reshape(in[1]->shape);
	return true;
}

template <typename F>
void AdditionOperation<F>::forward(Tensor<F> &in1, Tensor<F> &in2, Tensor<F> &out) {
  F alpha(1);

  add_cuda<F>(in1.ptr(), out.ptr(), in1.size(), 1.0);
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
void SigmoidOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta) {
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
void ReluOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta) {
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
void SoftmaxOperation<F>::backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta) {
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

template struct InputOperation<float>;
template struct ConvolutionOperation<float>;
template struct ConvolutionTransposeOperation<float>;
template struct LocalNormalisationOperation<float>;
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
template struct SquaredLossOperation<float>;


template struct InputOperation<double>;
template struct ConvolutionOperation<double>;
template struct ConvolutionTransposeOperation<double>;
template struct LocalNormalisationOperation<double>;
template struct SquashOperation<double>;
template struct UnsquashOperation<double>;
template struct MergeOperation<double>;
template struct SplitOperation<double>;
template struct AdditionOperation<double>;
template struct PoolingOperation<double>;
template struct TanhOperation<double>;
template struct SigmoidOperation<double>;
template struct ReluOperation<double>;
template struct SoftmaxOperation<double>;
template struct SquaredLossOperation<double>;
