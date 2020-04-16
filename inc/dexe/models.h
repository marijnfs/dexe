#pragma once

#include "network.h"
#include <functional>

namespace dexe {
    template <typename F> DEXE_API
        std::function<Node<F>(Node<F>)> basic_layer(int c, int c_out, int k);
    

    template <typename F> DEXE_API
        Node<F> make_unet(Network<F> *network, int in_channels, int out_channels, bool local_normalization = false);
}
