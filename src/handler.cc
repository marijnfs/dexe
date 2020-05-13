#include "dexe/handler.h"
#include "dexe/util.h"
#include "dexe/allocator.h"

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

    handle_error(curandCreateGenerator(&h_curand, CURAND_RNG_PSEUDO_DEFAULT));
    handle_error(curandSetPseudoRandomGeneratorSeed(h_curand, 13131ULL));
    handle_error(cublasCreate(&h_cublas));
    init_allocator();
}

void Handler::deinit() {
    if (!s_handler)
        return;
    
    clear_workspace();
    
    if (s_handler->h_cudnn) {
        cerr << "Destroying Cudnn" << endl;
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

    if (s_handler->one_float_) {
        handle_error( cudaFree(s_handler->one_float_) );
        s_handler->one_float_ = 0;
    }

    if (s_handler->one_double_) {
        handle_error( cudaFree(s_handler->one_double_) );
        s_handler->one_double_ = 0;
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

float *Handler::one_float() {
    auto &h = get_handler();
    if (!h.one_float_)
        cudaMalloc((void **)&h.one_float_, sizeof(float));

    float one(1);
    handle_error( cudaMemcpy(h.one_float_, &one, sizeof(float), cudaMemcpyHostToDevice) );
    return h.one_float_;
}

double *Handler::one_double() {
    auto &h = get_handler();
    if (!h.one_double_)
        cudaMalloc((void **)&h.one_double_, sizeof(double));

    double one(1);
    handle_error( cudaMemcpy(h.one_double_, &one, sizeof(double), cudaMemcpyHostToDevice) );
    return h.one_double_;
}



char *Handler::workspace() {
    auto &h = get_handler();
    if (!h.s_workspace) {
        handle_error(cudaMalloc((void **)&h.s_workspace, h.workspace_size_));
    }

    return h.s_workspace;
}

size_t Handler::workspace_size() {
    return get_handler().workspace_size_;
}

void Handler::set_workspace_size(size_t workspace_size) {
    clear_workspace();
    get_handler().workspace_size_ = workspace_size;
}

void Handler::clear_workspace() {
    auto &h = get_handler();
    if (h.s_workspace) {
        handle_error(cudaFree(h.s_workspace));
        h.s_workspace = nullptr;
    }
}

void Handler::init_allocator() {
    auto &h = get_handler();

    if (h.allocator_stack.empty())
        h.allocator_stack.push(new DirectAllocator());
}

void Handler::push_allocator(Allocator *allocator) {
    auto &h = get_handler();
    h.allocator_stack.push(allocator);
    // std::cout << "Dexe: Push Allocator" << std::endl;
}

void Handler::pop_allocator() {
    auto &h = get_handler();
    h.allocator_stack.pop();
    // std::cout << "Dexe: Pop Allocator" << std::endl;
}

Allocator *Handler::get_allocator() {
    auto &h = get_handler();
    return h.allocator_stack.top();
}


void Handler::print_mem_info() {
    size_t free_mem(0), total_mem(0);
    cudaMemGetInfo ( &free_mem, &total_mem );

    cout << "GPU Memory Free: " << free_mem << " Total: " << total_mem << endl;
}

void Handler::sync() {
    handle_error( cudaDeviceSynchronize() );
}

} // namespace dexe
