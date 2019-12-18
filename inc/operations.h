#pragma once

#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>
#include <string>
#include "external/cereal/archives/portable_binary.hpp"

#include "tensor.h"
#include "util.h"
#include "cudaptr.h"

// int const CONV_MAX_MEM = 0;
int const CONV_MAX_MEM = 1024 * 1024 * 1024;

namespace dexe {

template <typename F>
struct Operation {
	// virtual void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0){}

	// virtual void backward_weights(Tensor<F> &in, Tensor<F> &out_grad, F beta = 0.0){}
	// virtual void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0){}

	virtual TensorShape output_shape(TensorShape input) { return input; }

	// Runs the forward step
	virtual void forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) { throw std::runtime_error("Not Implemented"); }

	// Responsible for both checking if sizes match, and making sure the memory is allocated
	virtual bool forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out){ throw std::runtime_error("Not Implemented"); }


	// Runs the backward step
	virtual void backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad) { throw std::runtime_error("Not Implemented"); }
    virtual bool backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad) { throw std::runtime_error("Not Implemented"); }
	
    // Write a readable string to the ostream
	virtual void describe(std::ostream &out){}

	virtual OperationCode opcode() { throw std::runtime_error("Not Implemented"); }

	virtual void save(cereal::PortableBinaryOutputArchive &ar) {throw std::runtime_error("Not Implemented"); }
	
	// virtual void forward_timed(Tensor<F> &in, Tensor<F> &out, int t, F beta = 0.0){ forward(in, out, beta); }
	// virtual void backward_weights_timed(Tensor<F> &in, Tensor<F> &out_grad, int t, F beta = 0.0){}
	// virtual void backward_timed(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, int t, F beta = 0.0){}
};

template <typename F>
struct DefaultOperation : public Operation<F> {
	void forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) { forward(*in[0], *out[0]); }
	void backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad) { backward(*in[0], *out[0], *in_grad[0], *out_grad[0]); }
	bool forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out) { out[0]->reshape(output_shape(in[0]->shape) ); return true; }
    bool backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_) { in_grad[0]->reshape(in[0]->shape); return true; }

    virtual TensorShape output_shape(TensorShape input) = 0;
	virtual void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0) = 0;
	virtual void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0) = 0;
};

template <typename F>
struct Parametrised {
	virtual void init_normal(F mean, F std) = 0;
	virtual void init_uniform(F var) = 0;
	virtual void update(F lr) {}
	virtual void l2(F l) {}
	virtual void zero_grad() {}
	virtual void scale_grad(float val) {}
	virtual void register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &fast_params, std::vector<CudaPtr<F>> &grads, std::vector<CudaPtr<F> > &fast_grads) {}

	virtual std::vector<F> to_vector() { return std::vector<F>(); }
	virtual void from_vector(std::vector<F> &v) { }
	virtual int size() { return 0; }
	virtual std::vector<F> grad_to_vector() { return std::vector<F>(); }
};

template <typename F>
struct InputOperation : public Operation<F> {
    Tensor<F> *reference = nullptr;
    
    InputOperation(int n_channels_, Tensor<F> *reference_ = nullptr) : n_channels(n_channels_), reference(reference_) {}
	InputOperation(cereal::PortableBinaryInputArchive &ar);

    bool forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
    void forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
	void backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
    bool backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
	OperationCode opcode() { return INPUT; }

	void save(cereal::PortableBinaryOutputArchive &ar);
    int n_channels = 0;
};

template <typename F>
struct ConvolutionOperation : public Operation<F>, public Parametrised<F> {
	ConvolutionOperation(std::vector<int> dimensions, std::vector<int> strides, bool keep_, bool has_bias = true, size_t workspace_limit_ = CONV_MAX_MEM);
	ConvolutionOperation(cereal::PortableBinaryInputArchive &ar);

	~ConvolutionOperation();

	virtual void init_normal(F mean, F std);
	virtual void init_uniform(F var);
	void init();

	bool check_fit(Tensor<F> &in_tensor, Tensor<F> &out_tensor);

    // API
	virtual void forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
	virtual bool forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
    virtual bool backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
	virtual void backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
	virtual OperationCode opcode() { return CONVOLUTION; }
	virtual void save(cereal::PortableBinaryOutputArchive &ar);

    // regular
	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0);

    void prepare_forward(Tensor<F> &in, Tensor<F> &out);
    void prepare_backward_weights(Tensor<F> &in, Tensor<F> &out);
    void prepare_backward(Tensor<F> &in, Tensor<F> &out);
	void backward_weights(Tensor<F> &in, Tensor<F> &out_grad, F beta = 0.0);

	void update(F lr);
	void l2(F l);
	void zero_grad();
	void scale_grad(F val);
	void register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &fast_params, std::vector<CudaPtr<F>> &grads, std::vector<CudaPtr<F> > &fast_grads) override;
	void share(ConvolutionOperation<F> &other);



	std::vector<F> to_vector();
	void from_vector(std::vector<F> &v);
	std::vector<F> grad_to_vector();
	virtual int size();

	TensorShape output_shape(TensorShape input);
	void describe(std::ostream &out) { out << filter_bank.dimensions; }

    std::vector<int> dimensions, strides, paddings, dilations;

	cudnnConvolutionDescriptor_t conv = nullptr;
	FilterBank<F> filter_bank, filter_bank_grad;
	Tensor<F> bias, bias_grad;

	cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	cudnnConvolutionBwdDataAlgo_t algo_bwd = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
	cudnnConvolutionBwdFilterAlgo_t algo_bwd_filter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

	char *workspace = nullptr;
	char *workspace_bwd = nullptr;
	char *workspace_bwd_filter = nullptr;

	size_t workspace_size = 0;
	size_t workspace_size_bwd = 0;
	size_t workspace_size_bwd_filter = 0;

	bool has_bias = true;
	bool keep = true;
};

template <typename F>
struct ConvolutionTransposeOperation : public ConvolutionOperation<F> {
	ConvolutionTransposeOperation(std::vector<int> dimensions, std::vector<int> strides, bool keep_, size_t workspace_limit_ = CONV_MAX_MEM);
	ConvolutionTransposeOperation(cereal::PortableBinaryInputArchive &ar);

    // API
	virtual void forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
	virtual bool forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
    virtual bool backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
	virtual void backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);

	virtual void save(cereal::PortableBinaryOutputArchive &ar);

    virtual OperationCode opcode() { return CONVOLUTION_TRANSPOSE; }
};

template <typename F>
struct SquaredLossOperation : public Operation<F> {
  SquaredLossOperation();

  virtual void forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
  virtual bool forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
  virtual bool backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
  virtual void backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
  virtual OperationCode opcode() { return SQUARED_LOSS; }
	void save(cereal::PortableBinaryOutputArchive &ar){}

  void describe(std::ostream &out) { out << "squared_loss"; }

  Tensor<F> tmp; // temporary tensor for computation
};

template <typename F>
struct SupportLossOperation : public Operation<F> {
  SupportLossOperation(F support);
	SupportLossOperation(cereal::PortableBinaryInputArchive &ar);

  virtual void forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
  virtual bool forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
  virtual bool backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
  virtual void backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
  virtual OperationCode opcode() { return SUPPORT_LOSS; }
  void save(cereal::PortableBinaryOutputArchive &ar);

  void describe(std::ostream &out) { out << "support_loss"; }

  F support = 0;
  Tensor<F> tmp; // temporary tensor for computation
};

template <typename F>
struct SquashOperation : public ConvolutionOperation<F> {
	SquashOperation(TensorShape s, int c);
	TensorShape output_shape(TensorShape input);

	int c;

	void describe(std::ostream &out) { out << "squash"; }
    void init_normal(F mean, F std);
    void init_uniform(F var);

};

template <typename F>
struct UnsquashOperation : public Operation<F> {
  UnsquashOperation(TensorShape s);

  TensorShape output_shape(TensorShape input);
  void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
  void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0);
  
  void describe(std::ostream &out) { out << "unsquash"; }
  
  TensorShape s;
};

template <typename F>
struct MergeOperation : public Operation<F> {
  MergeOperation();

  TensorShape output_shape(TensorShape input);
  void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
  void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0);
  
  void describe(std::ostream &out) { out << "merge"; }
};

template <typename F>
struct SplitOperation : public Operation<F> {
  SplitOperation();

  TensorShape output_shape(TensorShape input);
  void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
  void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0);
  
  void describe(std::ostream &out) { out << "split"; }
};


template <typename F>
struct LocalNormalisationOperation : public DefaultOperation<F> {
  LocalNormalisationOperation(int w);
   LocalNormalisationOperation(cereal::PortableBinaryInputArchive &ar);

  TensorShape output_shape(TensorShape input);
  void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
  void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0);
  
  void describe(std::ostream &out) { out << "local_normalisation"; }
  virtual OperationCode opcode() { return LOCAL_NORMALISATION; }
	void save(cereal::PortableBinaryOutputArchive &ar);
	cudnnLRNDescriptor_t lrn_desc = 0;
	
	int w = 0;
};


template <typename F>
struct PoolingOperation : public Operation<F> {
  PoolingOperation(int kw, int kh, cudnnPoolingMode_t mode = CUDNN_POOLING_MAX);
	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "pool " << kw << "x" << kh; }

	TensorShape output_shape(TensorShape input);

	int kw, kh;
	cudnnPoolingDescriptor_t pool;
};

template <typename F>
struct TanhOperation : public DefaultOperation<F> {
  TanhOperation(F scale = 1.0);
	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "tanh"; }
	virtual OperationCode opcode() { return TANH; }

	void save(cereal::PortableBinaryOutputArchive &ar) {}

	TensorShape output_shape(TensorShape input);

	cudnnActivationDescriptor_t desc;

	F scale;
};

template <typename F>
struct SigmoidOperation : public DefaultOperation<F> {
  SigmoidOperation(F scale = 1.0);
	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "sigmoid"; }
	virtual OperationCode opcode() { return SIGMOID; }
	void save(cereal::PortableBinaryOutputArchive &ar) {}

	TensorShape output_shape(TensorShape input);
	cudnnActivationDescriptor_t desc;

	F scale;
};

template <typename F>
struct AdditionOperation : public Operation<F> {
  AdditionOperation();

	void forward(Tensor<F> &in1, Tensor<F> &in2, Tensor<F> &out);
	void backward(Tensor<F> &in_grad, Tensor<F> &out_grad1, Tensor<F> &in_grad2);
	void describe(std::ostream &out) { out << "addition"; }
	virtual OperationCode opcode() { return ADDITION; }

	void forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
	bool forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);
	void backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
    bool backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
	void save(cereal::PortableBinaryOutputArchive &ar) {}

	TensorShape output_shape(TensorShape input);
	cudnnActivationDescriptor_t desc;
};


template <typename F>
struct ReluOperation : public DefaultOperation<F> {
    ReluOperation();

	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "relu"; }
	virtual OperationCode opcode() { return RELU; }
	void save(cereal::PortableBinaryOutputArchive &ar) {}

	TensorShape output_shape(TensorShape input);
	cudnnActivationDescriptor_t desc;

};


template <typename F>
struct SoftmaxOperation : public DefaultOperation<F> {
	SoftmaxOperation(bool matched = false);
	void forward(Tensor<F> &in, Tensor<F> &out, F beta = 0.0);
	void backward(Tensor<F> &in, Tensor<F> &out, Tensor<F> &in_grad, Tensor<F> &out_grad, F beta = 0.0);
	void describe(std::ostream &out) { out << "softmax"; }
	virtual OperationCode opcode() { return SOFTMAX; }
	void save(cereal::PortableBinaryOutputArchive &ar) {}

	TensorShape output_shape(TensorShape input);
	bool matched;
};

template <typename F>
struct InstanceNormalisationOperation : public Operation<F> {
	InstanceNormalisationOperation();

	// Runs the forward step
	void forward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);

	// Responsible for both checking if sizes match, and making sure the memory is allocated
	bool forward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out);

	// Runs the backward step
	void backward(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
    bool backward_dry_run(std::vector<Tensor<F>*> &in, std::vector<Tensor<F>*> &out, std::vector<Tensor<F>*> &in_grad, std::vector<Tensor<F>*> &out_grad);
	
    // Write a readable string to the ostream
	void describe(std::ostream &out);

	OperationCode opcode() { return INSTANCE_NORMALISATION; }

	void save(cereal::PortableBinaryOutputArchive &ar);

	Tensor<F> tmp;
	F variance = 0.0;
};

}
