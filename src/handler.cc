#include "dexe/handler.h"
#include "dexe/util.h"

using namespace std;

namespace dexe {

Handler *Handler::s_handler = 0;

Handler::Handler() : h_cudnn(0), h_cublas(0), h_curand(0), s_workspace(0) {}

Handler::~Handler() {
    // todo: delete stuff
    cudaDeviceSynchronize();
}

void Handler::init() {
    handle_error(cudnnCreate(&h_cudnn));

    // handle_error( curandCreateGenerator(&h_curand,
    // CURAND_RNG_PSEUDO_XORWOW));
    handle_error(curandCreateGenerator(&h_curand, CURAND_RNG_PSEUDO_DEFAULT));
    handle_error(curandSetPseudoRandomGeneratorSeed(h_curand, 13131ULL));
    // handle_error( curandSetQuasiRandomGeneratorDimensions(h_curand, 1) );
    handle_error(cublasCreate(&h_cublas));

    handle_error(cudaMalloc((void **)&s_workspace, WORKSPACE_SIZE));
}

void Handler::deinit() {
    if (!s_handler)
        return;
    if (s_handler->s_workspace) {
        handle_error(cudaFree(s_handler->s_workspace));
        s_handler->s_workspace = 0;
    }
    if (s_handler->h_cudnn) {
        cerr << "destroying cudnn" << endl;
        handle_error(cudnnDestroy(s_handler->h_cudnn));
        s_handler->h_cudnn = 0;
    }
    if (s_handler->h_curand) {
        curandDestroyGenerator(s_handler->h_curand);
        s_handler->h_curand = 0;
    }
    if (s_handler->h_cublas) {
        cublasDestroy(s_handler->h_cublas);
        s_handler->h_cublas = 0;
    }

    delete s_handler;
    s_handler = 0;
}

void Handler::set_device(int n) {
    if (s_handler) {
        cerr << "Warning, setting device after Handler was initialized. Some "
                "handles do not deal well with that."
             << endl;
    }

    handle_error(cudaSetDevice(n));
    cudaDeviceProp prop;
    handle_error(cudaGetDeviceProperties(&prop, n));
    cout << "CuDNN Device Number: " << n << endl;
    cout << "Device name: " << prop.name;
}
    
Handler &Handler::get_handler() {
    if (!s_handler) {
        s_handler = new Handler();
        s_handler->init();    
    }
    return *s_handler;
}


cudnnHandle_t &Handler::cudnn() {
    return get_handler().h_cudnn;
}

curandGenerator_t &Handler::curand() {
    return get_handler().h_curand;
}

cublasHandle_t &Handler::cublas() {
    return get_handler().h_cublas;
}

char *Handler::workspace() {
    return get_handler().s_workspace;
}

} // namespace dexe
