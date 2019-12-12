include("SelectLibraryConfigurations")
include("FindPackageHandleStandardArgs")

find_path(CUDNN_INCLUDE_DIR cudnn.h)

find_library(CUDNN_LIBRARY_RELEASE NAMES cudnn cudnn7_64 PATH_SUFFIXES Release HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDNN_LIBRARY_DEBUG NAMES cudnn cudnn7_64 PATH_SUFFIXES Debug HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

select_library_configurations(CUDNN)
mark_as_advanced(CUDNN_LIBRARY_RELEASE CUDNN_LIBRARY_DEBUG)

find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_LIBRARIES CUDNN_INCLUDE_DIR)
mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARIES)