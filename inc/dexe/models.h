#pragma once

#include "network.h"
#include <functional>

namespace dexe {
    template <typename F>
        std::function<Node<F>(Node<F>)> basic_layer(int c, int c_out, int k);
    
    template <typename F>
        Node<F> make_unet(Network<F> *network, int in_channels, int out_channels);
}
