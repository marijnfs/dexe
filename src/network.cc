#include "dexe/network.h"
#include "dexe/operations.h"
#include "dexe/util.h"
#include "dexe/allocator.h"

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/set.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stack>
#include <vector>

using namespace std;

namespace dexe {

template <typename F> TensorShape Node<F>::shape() {
    return network->tensors[index].shape();
}

template <typename F> TensorSet<F> &Node<F>::tensor_set() {
    return network->tensors[index];
}

template <typename F>
void Node<F>::operator()(
    std::initializer_list<std::reference_wrapper<Tensor<F>>> input_tensors) {
    network->set_inputs(input_tensors);
    network->forward(network->inputs, {index});
}

template <typename F> void Node<F>::set_x(Tensor<F> &x) {
    network->tensors[index].x->reshape(x.shape);
    network->tensors[index].x->from_tensor(x);
}

template <typename F> 
Tensor<F> &Node<F>::x() {
    return *network->tensors[index].x;
}

template <typename F> 
Tensor<F> &Node<F>::grad() {
    return *network->tensors[index].grad;
}

template <typename F> Network<F>::Network() {}

template <typename F> void Network<F>::reset() {
    sequence.clear();
    names.clear();
    operations.clear();
    tensors.clear();
    input_indices.clear(); // inputs to every node

    inputs.clear(); // indices to nodes that function as inputs

    parameters.clear();

    param_vec.allocate(0);
    grad_vec.allocate(0);
    param_ptrs.clear();
    grad_ptrs.clear();

    fast_param_ptrs.clear();
    fast_grad_ptrs.clear();

    names_set.clear();

    n_params = 0;
    finished = false;
}

template <typename F> void Network<F>::empty_tensors(bool skip_input) {
    set<int> input_indices_set(inputs.begin(), inputs.end());

    int s(0);
    for (auto &tensor_set : tensors) {
        if (skip_input && input_indices_set.count(s))
            continue;
        auto shape = tensor_set.x->shape.zero_lowerdims();
        tensor_set.x->reshape(shape);
        tensor_set.grad->reshape(shape);
        ++s;
    }
}


template <typename F> void Network<F>::set_inputs(std::initializer_list<std::reference_wrapper<Tensor<F>>> input_tensors) {
    if (input_tensors.size() != inputs.size()) {
        cerr << "Warning: Number of inputs doesn't correspond";
    }

    auto input_it = inputs.begin();

    for (auto &input_tensor : input_tensors) {
        auto idx = *input_it;
        tensors[idx].x->reshape(input_tensor.get().shape);
        tensors[idx].x->from_tensor(input_tensor);
        ++input_it;
    }
}

template <typename F> void Network<F>::finish() {
    if (!finished)
        align_params();
}

template <typename F> void Network<F>::assert_finished() {
    if (!finished)
        throw DexeException("call network.finish() before using network");
}

template <typename F> void Network<F>::zero_x() {
    for (auto &tensor : tensors)
        if (tensor.x)
            tensor.x->zero();
}

template <typename F> void Network<F>::zero_grad() {
    for (auto &tensor : tensors)
        if (tensor.grad)
            tensor.grad->zero();

    for (auto &param : parameters)
        param->zero_grad();
}

template <typename F> void Network<F>::update(F lr) {
    assert_finished();
    for (size_t i(0); i < parameters.size(); ++i)
        parameters[i]->update(lr);
}

template <typename F> void Network<F>::l2(F l) {
    assert_finished();
    for (size_t i(0); i < parameters.size(); ++i)
        parameters[i]->l2(l);
}

template <typename F> void Network<F>::init_normal(F mean, F std) {
    finish();
    for (size_t i(0); i < parameters.size(); ++i)
        parameters[i]->init_normal(mean, std);
}

template <typename F> void Network<F>::init_uniform(F var) {
    finish();
    for (size_t i(0); i < parameters.size(); ++i)
        parameters[i]->init_uniform(var);
}

template <typename F> void Network<F>::save(std::string path) {
    ofstream of(path, ios::binary);
    cereal::PortableBinaryOutputArchive ar(of);

    ar(sequence);
    ar(names);
    ar(input_indices);
    ar(inputs);

    std::vector<TensorShape> shapes;
    for (auto &t : tensors)
        shapes.emplace_back(t.x->shape);
    ar(shapes);

    vector<OperationCode> opcodes;
    for (auto &op : operations)
        opcodes.emplace_back(op->opcode());
    ar(opcodes);

    for (auto &op : operations) {
        op->save(ar);
    }
}

template <typename F>

void Network<F>::load(std::string path) {
    ifstream in(path, ios::binary);
    cereal::PortableBinaryInputArchive ar(in);

    // reset current state
    reset();

    ar(sequence);
    ar(names);
    ar(input_indices);
    ar(inputs);
    std::vector<TensorShape> shapes;
    ar(shapes);
    for (auto shape : shapes)
        tensors.emplace_back(shape);

    vector<OperationCode> opcodes;
    ar(opcodes);

    for (auto opcode : opcodes) {
        Operation<F> *op = nullptr;

        if (opcode == INPUT) {
            op = new InputOperation<F>(ar);
        } else if (opcode == CONVOLUTION) {
            op = new ConvolutionOperation<F>(ar);
        } else if (opcode == CONVOLUTION_TRANSPOSE) {
            op = new ConvolutionTransposeOperation<F>(ar);
        } else if (opcode == SQUARED_LOSS) {
            op = new SquaredLossOperation<F>();
        } else if (opcode == SUPPORT_LOSS) {
            op = new SupportLossOperation<F>(ar);
        } else if (opcode == DICE_LOSS) {
            op = new DiceLossOperation<F>(ar);
        } else if (opcode == LOCAL_NORMALISATION) {
            op = new LocalNormalisationOperation<F>(ar);
        } else if (opcode == INSTANCE_NORMALISATION) {
            op = new InstanceNormalisationOperation<F>();
        } else if (opcode == TANH) {
            op = new TanhOperation<F>();
        } else if (opcode == SIGMOID) {
            op = new SigmoidOperation<F>();
        } else if (opcode == TANH) {
            op = new TanhOperation<F>();
        } else if (opcode == ADDITION) {
            op = new AdditionOperation<F>();
        } else if (opcode == RELU) {
            op = new ReluOperation<F>();
        } else if (opcode == SOFTMAX) {
            op = new SoftmaxOperation<F>();
        } else {
            throw std::runtime_error("Opcode not implemented");
        }

        operations.emplace_back(op);

        if (auto param = dynamic_cast<Parametrised<F> *>(op))
            parameters.emplace_back(param);
    }
    finish();
}

template <typename F> vector<F> Network<F>::to_vector() {
    vector<F> full_vec;
    for (size_t i(0); i < parameters.size(); ++i) {
        vector<F> vec = parameters[i]->to_vector();
        copy(vec.begin(), vec.end(), back_inserter(full_vec));
    }
    return full_vec;
}

template <typename F> void Network<F>::from_vector(vector<F> &vec) {
    typename vector<F>::iterator it(vec.begin());
    for (size_t i(0); i < parameters.size(); ++i) {
        vector<F> v(it, it + parameters[i]->size());
        parameters[i]->from_vector(v);
        it += parameters[i]->size();
    }
}

template <typename F> vector<F> Network<F>::fd_gradient(F e) {
    vector<F> full_grad;

    vector<int> outputs({int(operations.size()) - 1});

    for (size_t i(0); i < parameters.size(); ++i) {
        vector<F> vec = parameters[i]->to_vector();

        vector<F> delta_vec(vec);
        for (size_t n(0); n < vec.size(); ++n) {
            delta_vec[n] = vec[n] + e;
            parameters[i]->from_vector(delta_vec);

            forward(inputs, outputs);
            // forward_nograd(inputs, outputs);
            F plus_loss = tensors.back().x->to_vector()[0];

            delta_vec[n] = vec[n] - e;
            parameters[i]->from_vector(delta_vec);

            forward(inputs, outputs);
            // forward_nograd(inputs, outputs);

            F min_loss = tensors.back().x->to_vector()[0];

            full_grad.push_back((plus_loss - min_loss) / (2 * e));
            delta_vec[n] = vec[n];
        }
        parameters[i]->from_vector(vec);
    }
    return full_grad;
}

template <typename F> vector<F> Network<F>::gradient() {
    vector<F> full_grad;
    for (size_t i(0); i < parameters.size(); ++i) {
        vector<F> grad = parameters[i]->grad_to_vector();
        copy(grad.begin(), grad.end(), back_inserter(full_grad));
    }
    return full_grad;
}

template <typename F> void Network<F>::describe(ostream &out) {
    for (auto &o : operations) {
        o->describe(out);
        out << endl;
    }
    out.flush();
}

template <typename F> void Network<F>::align_params() {
    register_params();
    param_vec.allocate(n_params);
    grad_vec.allocate(n_params);

    F *ptr = param_vec.data;
    for (auto &p : param_ptrs) {
        handle_error(cudaMemcpy(ptr, p->data, p->N * sizeof(F),
                                cudaMemcpyDeviceToDevice));
        auto N = p->N;
        p->free();
        p->data = ptr;
        p->N = N;
        p->own = false;

        ptr += N;
    }

    ptr = grad_vec.data;
    for (auto &g : grad_ptrs) {
        handle_error(cudaMemcpy(ptr, g->data, g->N * sizeof(F),
                                cudaMemcpyDeviceToDevice));
        auto N = g->N;
        g->free();
        g->data = ptr;
        g->N = N;
        g->own = false;

        ptr += N;
    }

    finished = true;
}

template <typename F> void Network<F>::register_params() {
    param_ptrs.clear();
    grad_ptrs.clear();
    fast_param_ptrs.clear();
    fast_grad_ptrs.clear();

    for (auto &p : parameters)
        p->register_params(param_ptrs, fast_param_ptrs, grad_ptrs,
                           fast_grad_ptrs);

    n_params = 0;
    for (auto &p : param_ptrs)
        n_params += p->N;
}

template <typename F> Network<F>::~Network() { }

template <typename F> string Network<F>::get_unique_name(string name) {
    int n(0);
    while (true) {
        ostringstream oss;
        if (!n)
            oss << name;
        else
            oss << name << "_" << n;
        if (!names_set.count(oss.str()))
            return oss.str();
        ++n;
    }
    throw std::runtime_error("unreachable");
}

template <typename F>
int Network<F>::add_operation(Operation<F> *op, vector<int> inputs,
                              TensorShape shape, string name) {
    finished = false;

    int index = names.size();

    auto unique_name = get_unique_name(name);

    names.emplace_back(unique_name);
    names_set.insert(unique_name);

    operations.emplace_back(op);
    tensors.emplace_back(shape);

    input_indices.emplace_back(inputs);
    if (auto param = dynamic_cast<Parametrised<F> *>(op))
        parameters.emplace_back(param);
    return index;
}

template <typename F>
Node<F> Network<F>::input_1D(int n_channels, std::string name) {
    auto index = add_operation(new InputOperation<F>(n_channels), vector<int>{},
                               TensorShape(0, n_channels, 0), name);
    inputs.emplace_back(index);
    return Node<F>(index, this);
}


template <typename F>
Node<F> Network<F>::input_2D(int n_channels, std::string name) {
    auto index = add_operation(new InputOperation<F>(n_channels), vector<int>{},
                               TensorShape(0, n_channels, 0, 0), name);
    inputs.emplace_back(index);
    return Node<F>(index, this);
}

template <typename F>
Node<F> Network<F>::input_3D(int n_channels, std::string name) {
    auto index = add_operation(new InputOperation<F>(n_channels), vector<int>{},
                               {0, n_channels, 0, 0, 0}, name);
    inputs.emplace_back(index);
    return Node<F>(index, this);
}

template <typename F> Node<F> Network<F>::get_node(std::string name) {
    for (int n(0); n < operations.size(); ++n)
        if (names[n] == name)
            return Node<F>(n, this);
    return Node<F>(-1, this);
}

template <typename F> Node<F> Network<F>::get_node(int index) {
    return Node<F>(index, this);
}

template <typename F> Node<F> Network<F>::get_last_node() {
    return Node<F>((int)operations.size() - 1, this);
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::convolution_1D(int out_c, int k,
                                                        string name) {
    return [this, out_c, k, name](Node<F> n) {
        auto in_c = n.shape().c();

        auto index = add_operation(
            new ConvolutionOperation<F>({out_c, in_c, k}, {1,}, true),
            vector<int>{n.index}, {0, out_c, 0}, name);
        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::convolution_2D(int out_c, int k,
                                                        string name) {
    return [this, out_c, k, name](Node<F> n) {
        auto in_c = n.shape().c();

        auto index = add_operation(
            new ConvolutionOperation<F>({out_c, in_c, k, k}, {1, 1}, true),
            vector<int>{n.index}, TensorShape{0, out_c, 0, 0}, name);
        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::convolution_3D(int out_c, int k,
                                                           string name) {
    return [this, out_c, k, name](Node<F> n) {
        auto in_c = n.shape().c();
        auto index = add_operation(new ConvolutionOperation<F>(
                                       {out_c, in_c, k, k, k}, {1, 1, 1}, true),
                                   vector<int>{n.index},
                                   TensorShape{0, out_c, 0, 0, 0}, name);
        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)>
Network<F>::convolution_downscale(int out_c, int k, string name) {
    return [this, out_c, k, name](Node<F> n) {
        auto in_c = n.shape().c();
        auto index = add_operation(
            new ConvolutionOperation<F>({out_c, in_c, k, k}, {k, k}, false),
            vector<int>{n.index}, TensorShape{0, out_c, 0, 0}, name);
        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)>
Network<F>::convolution_downscale_3D(int out_c, int k, string name) {
    return [this, out_c, k, name](Node<F> n) {
        auto in_c = n.shape().c();
        auto index = add_operation(
            new ConvolutionOperation<F>({out_c, in_c, k, k, k}, {k, k, k},
                                        false),
            vector<int>{n.index}, TensorShape{0, out_c, 0, 0, 0}, name);
        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)>
Network<F>::convolution_upscale(int out_c, int k, string name) {
    return [this, out_c, k, name](Node<F> n) {
        auto in_c = n.shape().c();
        // in Conv Transpose, the in_c and out_c ordering logic is reversed
        auto index = add_operation(new ConvolutionTransposeOperation<F>(
                                       {in_c, out_c, k, k}, {k, k}, false),
                                   vector<int>{n.index},
                                   TensorShape{0, out_c, 0, 0}, name);
        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)>
Network<F>::convolution_upscale_3D(int out_c, int k, string name) {
    return [this, out_c, k, name](Node<F> n) {
        auto in_c = n.shape().c();
        // in Conv Transpose, the in_c and out_c ordering logic is reversed
        auto index = add_operation(
            new ConvolutionTransposeOperation<F>({in_c, out_c, k, k, k},
                                                 {k, k, k}, false),
            vector<int>{n.index}, TensorShape{0, out_c, 0, 0, 0}, name);
        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::relu(string name) {
    return [this, name](Node<F> n) {
        auto in_c = n.shape().c();
        auto index = add_operation(new ReluOperation<F>(), vector<int>{n.index},
                                   TensorShape{0, in_c, 0, 0}, name);

        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::sigmoid(string name) {
    return [this, name](Node<F> n) {
        auto in_c = n.shape().c();
        auto index =
            add_operation(new SigmoidOperation<F>(), vector<int>{n.index},
                          TensorShape{0, in_c, 0, 0}, name);

        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::tanh(string name) {
    return [this, name](Node<F> n) {
        auto in_c = n.shape().c();
        auto index = add_operation(new TanhOperation<F>(), vector<int>{n.index},
                                   TensorShape{0, in_c, 0, 0}, name);

        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)> Network<F>::local_normalisation(int k,
                                                                string name) {
    return [this, name, k](Node<F> n) {
        auto in_c = n.shape().c();
        auto index = add_operation(new LocalNormalisationOperation<F>(k),
                                   vector<int>{n.index},
                                   TensorShape{0, in_c, 0, 0}, name);

        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)>
Network<F>::local_normalisation_3D(int k, string name) {
    return [this, name, k](Node<F> n) {
        auto in_c = n.shape().c();
        auto index = add_operation(new LocalNormalisationOperation<F>(k),
                                   vector<int>{n.index},
                                   TensorShape{0, in_c, 0, 0, 0}, name);

        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>)>
Network<F>::instance_normalisation(string name) {
    return [this, name](Node<F> n) {
        auto in_c = n.shape().c();
        auto index = add_operation(new InstanceNormalisationOperation<F>(),
                                   vector<int>{n.index},
                                   TensorShape{0, in_c, 0, 0, 0}, name);

        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>, Node<F>)>
Network<F>::squared_loss(std::string name) {
    return [this, name](Node<F> n1, Node<F> n2) {
        auto in_c = n1.shape().c();
        auto index = add_operation(new SquaredLossOperation<F>(),
                                   vector<int>{n1.index, n2.index},
                                   TensorShape{0, in_c, 0, 0, 0}, name);

        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>, Node<F>)>
Network<F>::support_loss(F support, std::string name) {
    return [this, name, support](Node<F> n1, Node<F> n2) {
        auto in_c = n1.shape().c();
        auto index = add_operation(new SupportLossOperation<F>(support),
                                   vector<int>{n1.index, n2.index},
                                   TensorShape{0, in_c, 0, 0, 0}, name);

        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>, Node<F>)>
Network<F>::dice_loss(F smoothing, std::string name) {
    return [this, name, smoothing](Node<F> n1, Node<F> n2) {
        auto in_c = n1.shape().c();
        auto index = add_operation(new DiceLossOperation<F>(smoothing),
                                   vector<int>{n1.index, n2.index},
                                   TensorShape{0, in_c, 0, 0, 0}, name);

        return Node<F>(index, this);
    };
}

template <typename F>
std::function<Node<F>(Node<F>, Node<F>)> Network<F>::addition(string name) {
    return [this, name](Node<F> n1, Node<F> n2) {
        auto in_c = n1.shape().c();
        auto index = add_operation(new AdditionOperation<F>(),
                                   vector<int>{n1.index, n2.index},
                                   TensorShape{0, in_c, 0, 0}, name);

        return Node<F>(index, this);
    };
}

template <typename F> void Network<F>::backward() {
    if (sequence.empty()) {
        cerr << "No sequence available, did you run forward?" << endl;
        return;
    }

    for (auto it = sequence.rbegin(); it != sequence.rend(); ++it) {
        int s = *it;

        vector<Tensor<F> *> tmp_inputs, tmp_outputs, tmp_input_grads,
            tmp_output_grads;
        for (auto idx : input_indices[s]) {
            tmp_inputs.push_back(tensors[idx].x.get());
            tmp_input_grads.push_back(tensors[idx].grad.get());
        }

        tmp_outputs.push_back(tensors[s].x.get());
        tmp_output_grads.push_back(tensors[s].grad.get());

        operations[s]->backward_dry_run(tmp_inputs, tmp_outputs,
                                        tmp_input_grads, tmp_output_grads);
        operations[s]->backward(tmp_inputs, tmp_outputs, tmp_input_grads,
                                tmp_output_grads);
    }
}

template <typename F>
vector<int> Network<F>::find_sequence(std::vector<int> inputs,
                                      std::vector<int> outputs) {
    vector<int> sequence;

    // First mark which nodes are possible active, to avoid calculating too far
    vector<bool> active(operations.size());
    {
        stack<int> node_stack;
        for (auto o : outputs)
            node_stack.push(o);

        while (!node_stack.empty()) {
            auto cur = node_stack.top();
            node_stack.pop();
            active[cur] = true;
            for (auto dep : input_indices[cur])
                node_stack.push(dep);
        }
    }

    for (auto i : inputs)
        if (!tensors[i].x) {
            cerr << "Input tensor at index " << i << " is not set" << endl;
            return vector<int>();
        }

    // Build the forward dependency (resolution) graph
    vector<vector<int>> output_indices(operations.size());
    for (int n(0); n < operations.size(); ++n)
        for (int in_n : input_indices[n])
            output_indices[in_n].push_back(n);

    stack<int> node_stack;
    set<int> resolved_nodes;

    for (auto i : inputs)
        node_stack.push(i);

    while (!node_stack.empty()) {
        auto cur = node_stack.top();
        bool resolved = true;
        for (auto n : output_indices[cur])
            if (!resolved_nodes.count(n) && active[n]) {
                node_stack.push(n);
                resolved = false;
            }

        if (!resolved)
            continue;

        node_stack.pop();
        resolved_nodes.insert(cur);

        // don't add leaf nodes to calculation sequence
        sequence.push_back(cur);
    }

    // put calculation nodes in order
    reverse(sequence.begin(), sequence.end());
    return sequence;
}

template <typename F>
void Network<F>::forward(std::vector<int> inputs, std::vector<int> outputs) {
    sequence = find_sequence(inputs, outputs);

    set<int> input_set(inputs.begin(), inputs.end());

    // Forward Dryrun
    for (auto s : sequence) {
        vector<Tensor<F> *> tmp_inputs, tmp_outputs;
        for (auto idx : input_indices[s]) {
            tmp_inputs.push_back(tensors[idx].x.get());
        }
        tmp_outputs.push_back(tensors[s].x.get());

        bool success = operations[s]->forward_dry_run(tmp_inputs, tmp_outputs);
        if (!success) {
            ostringstream oss;
            oss << "Failure when preparing step [" << s << "]: " << names[s] << endl;
            throw std::runtime_error(oss.str());
        }
    }

    // make sure buffers are zeroed out
    for (auto s : sequence)
        if (!input_set.count(s))
            tensors[s].x->zero();


    // Run Forward
    for (auto s : sequence) {
        vector<Tensor<F> *> tmp_inputs, tmp_outputs;
        for (auto idx : input_indices[s])
            tmp_inputs.push_back(tensors[idx].x.get());
        tmp_outputs.push_back(tensors[s].x.get());

        operations[s]->forward(tmp_inputs, tmp_outputs);
    }
}

template <typename F>
void Network<F>::forward_nograd(std::vector<int> inputs, std::vector<int> outputs) {
    sequence = find_sequence(inputs, outputs);

    set<int> input_set(inputs.begin(), inputs.end());

    //vector to indicate how many times an input is used, to know when we can release it
    vector<int> usage_counts(sequence.size());

    for (auto s : sequence)
        for (auto idx : input_indices[s])
            ++usage_counts[idx];

    // Forward Dryrun to determine memory usage
    auto virtual_allocator = std::make_unique<VirtualAllocator>();

    auto dry_run = [&](bool execute) {
        vector<int> usages_left(usage_counts);

        for (auto s : sequence) {
            vector<Tensor<F> *> tmp_inputs, tmp_outputs;
            for (auto idx : input_indices[s]) {
                tmp_inputs.push_back(tensors[idx].x.get());
            }
            tmp_outputs.push_back(tensors[s].x.get());

            cout << "dry run step " << s << endl;
            bool success = operations[s]->forward_dry_run(tmp_inputs, tmp_outputs);
            if (!success) {
                ostringstream oss;
                oss << "Failure when preparing step [" << s << "]: " << names[s] << endl;
                throw std::runtime_error(oss.str());
            }

            if (execute)
                operations[s]->forward(tmp_inputs, tmp_outputs);

            //decrease counter and free if needed
            for (auto idx : input_indices[s])
                if (--usages_left[idx] == 0 && input_set.count(idx) == 0)
                    tensors[idx].x->reshape(tensors[idx].x->shape.zero_lowerdims()); //reshaping to empty shape releases
        }
    };


    std::cout << "before dry run: " << std::endl;
    for (auto &t : tensors)
        std::cout << "tensor shape: " << t.x->shape << std::endl;

    push_allocator(virtual_allocator.get());
    dry_run(false);
    pop_allocator();

    size_t n_bytes = virtual_allocator->max_size();
    std::cout << "before dry run 2: " << std::endl;
    std::cout << "N Bytes: " << n_bytes << std::endl;

    for (auto &t : tensors)
        std::cout << "tensor shape: " << t.x->shape << std::endl;

    auto page_allocator = std::make_unique<MappedAllocator>(n_bytes);
    // push_allocator(page_allocator.get());
    dry_run(true);
    // pop_allocator();
}


template struct DEXE_API Node<float>;
template struct DEXE_API Node<double>;

template struct DEXE_API Network<float>;
template struct DEXE_API Network<double>;

} // namespace dexe
