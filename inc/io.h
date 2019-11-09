#pragma once

#include <iostream>
#include <nifti1_io.h>
#include "tensor.h"

template <typename F>
std::unique_ptr<Tensor<F>> read_nifti(std::string &path) {
    auto image = nifti_image_read(path.c_str(), 1);
    if (!image)
    	throw std::runtime_error("reading failed");
    
    auto dtype = image->datatype;
    auto n_dim = image->dim[0];
    auto bytesper = image->nbyper;
    auto n_vox = image->nvox;

    std::string description(image->descrip);
    
    std::vector<int> dims(n_dim + 2);
    dims[0] = dims[1] = 1;
    std::copy(&image->dim[1], &image->dim[1+n_dim], &dims[2]);

    std::cout << dims << std::endl;
    std::cout << description << std::endl;
    std::cout << n_vox << " x " << bytesper << std::endl;
    
    std::vector<F> converted(n_vox);
    if (bytesper == 2) {
        uint16_t *data = (uint16_t*) image->data;
        std::copy(data, data + n_vox, converted.begin());
    }
    auto tensor = std::make_unique<Tensor<F>>(TensorShape{dims});
    tensor->from_vector(converted);
    return tensor;
}

template std::unique_ptr<Tensor<float>> read_nifti(std::string &path);
template std::unique_ptr<Tensor<double>> read_nifti(std::string &path);
