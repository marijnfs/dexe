#include "dexe/network.h"

#include <functional>

using std::function;

namespace dexe {

template <typename F>
function<Node<F>(Node<F>)> basic_layer(int c, int c_out, int k) {
    return [c, c_out, k](Node<F> node) {
        auto network = node.network;
        if (!network)
            throw std::runtime_error("basic_layer: No network in node");
        auto node_last = network->convolution_3D(c_out, k)(node);
        node = network->convolution_3D(c, k)(node);
        node = network->relu()(node);
        node = network->convolution_3D(c_out, k)(node);
        node = network->addition()(node_last, node);

        // node = network->relu()(node);
        return node;
    };
}

template <typename F>
Node<F> make_unet(Network<F> *network, int in_channels, int out_channels, bool local_normalization) {
    int c = 2;
    int k = 3;
    int norm_k = 5;

    auto in = network->input_3D(in_channels);
    if (local_normalization)
        in = network->local_normalisation_3D(norm_k)(in);

    auto l0 = basic_layer<F>(c, c, k)(in);

    int c1 = c << 1;
    auto l1 = network->convolution_downscale_3D(c1, 2)(l0);
    l1 = basic_layer<F>(c1, c1, k)(l1);

    int c2 = c << 2;
    auto l2 = network->convolution_downscale_3D(c2, 2)(l1);
    l2 = basic_layer<F>(c2, c2, k)(l2);

    int c3 = c << 3;
    auto l3 = network->convolution_downscale_3D(c3, 2)(l2);
    l3 = basic_layer<F>(c3, c3, k)(l3);

    auto l2_down = network->convolution_upscale_3D(c2, 2)(l3);
    l2_down = basic_layer<F>(c2, c2, k)(l2_down);
    l2 = network->addition()(l2, l2_down);

    auto l1_down = network->convolution_upscale_3D(c1, 2)(l2_down);
    l1_down = basic_layer<F>(c1, c1, k)(l1_down);
    l1 = network->addition()(l1, l1_down);

    auto l0_down = network->convolution_upscale_3D(c, 2)(l1_down);
    l0_down = basic_layer<F>(c, c, k)(l0_down);
    l0 = network->addition()(l0, l0_down);

    auto prediction = network->convolution_3D(out_channels, k)(l0);
    return prediction;
}

template Node<float> DEXE_API make_unet(Network<float> *network, int in_channels,
                               int out_channels, bool local_normalization);
template Node<double> DEXE_API make_unet(Network<double> *network, int in_channels,
                                int out_channels, bool local_normalization);

} // namespace dexe
