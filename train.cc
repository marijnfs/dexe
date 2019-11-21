#include "network.h"

#include "tensor.h"
#include <vector>
#include <thread>

using namespace std;
using namespace dexe;

typedef function<void(vector<unique_ptr<Tensor<float>>>*)> GrabSamplesFunc;

struct Trainer {
    Trainer(GrabSamplesFunc func) : grab_samples_func(func) {}


    void start() {
        grab_samples_func(&samples);

        while (true) {
            thread t(grab_samples_func, &next_samples);
            
        }
    }

    vector<unique_ptr<Tensor<float>>> samples;
    vector<unique_ptr<Tensor<float>>> next_samples;

    GrabSamplesFunc grab_samples_func;
};

int main(int argc, char **argv) {
    Network<float> network;

    
}
