#include "dexe/io.h"


#include <nifti/nifti1_io.h>

namespace dexe {

std::unique_ptr<Tensor<float>> read_nifti(std::string &path) {
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
    std::cout << n_vox << " x " << bytesper << " dtype: " << nifti_datatype_to_string(dtype) << std::endl;
    
    std::vector<float> converted(n_vox);

    auto data = image->data;
    if (dtype == NIFTI_TYPE_UINT8)
        std::copy(reinterpret_cast<uint8_t*>(data), reinterpret_cast<uint8_t*>(data) + n_vox, converted.begin());
    else if (dtype == NIFTI_TYPE_INT16)
        std::copy(reinterpret_cast<int16_t*>(data), reinterpret_cast<int16_t*>(data) + n_vox, converted.begin());
    else if (dtype == NIFTI_TYPE_INT32)
        std::copy(reinterpret_cast<int32_t*>(data), reinterpret_cast<int32_t*>(data) + n_vox, converted.begin());
    else if (dtype == NIFTI_TYPE_FLOAT32)
        std::copy(reinterpret_cast<float*>(data), reinterpret_cast<float*>(data) + n_vox, converted.begin());
    else if (dtype == NIFTI_TYPE_FLOAT64)
        std::copy(reinterpret_cast<double*>(data), reinterpret_cast<double*>(data) + n_vox, converted.begin());
    else if (dtype == NIFTI_TYPE_INT8)
        std::copy(reinterpret_cast<int8_t*>(data), reinterpret_cast<int8_t*>(data) + n_vox, converted.begin());
    else if (dtype == NIFTI_TYPE_UINT16)
        std::copy(reinterpret_cast<uint16_t*>(data), reinterpret_cast<uint16_t*>(data) + n_vox, converted.begin());
    else if (dtype == NIFTI_TYPE_UINT32)
        std::copy(reinterpret_cast<uint32_t*>(data), reinterpret_cast<uint32_t*>(data) + n_vox, converted.begin());
    else if (dtype == NIFTI_TYPE_INT64)
        std::copy(reinterpret_cast<int64_t*>(data), reinterpret_cast<int64_t*>(data) + n_vox, converted.begin());
    else if (dtype == NIFTI_TYPE_UINT64)
        std::copy(reinterpret_cast<uint64_t*>(data), reinterpret_cast<uint64_t*>(data) + n_vox, converted.begin());
    else 
        throw std::runtime_error("reading failed, unsupported datatype");


    auto tensor = std::make_unique<Tensor<float>>(TensorShape{dims});
    tensor->from_vector(converted);
    return tensor;
}


}
