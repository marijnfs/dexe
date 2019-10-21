# CMakeLists files in this project can
# refer to the root source directory of the project as ${MCDNN_SOURCE_DIR} and

cmake_minimum_required (VERSION 2.8.11)
project (MCDNN)
#SET(CMAKE_BUILD_TYPE CACHE RELEASE STRING "Build Type")

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
	message(FATAL_ERROR "In-source builds are not permitted. Make a separate folder for building:\nmkdir build; cd build; cmake ..\nBefore that, remove the files already created:\nrm -rf CMakeCache.txt CMakeFiles")
endif()

find_package(CUDA QUIET REQUIRED)
find_library(CUDNN cudnn HINTS /usr/local/cuda/lib64 /usr/local/cuda/lib)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60 -std=c++11")

set(CMAKE_CXX_FLAGS "-std=c++11")

# Specify include directories
include_directories(
    inc
)

#add_subdirectory (inc)
#add_subdirectory (src)

file(GLOB CPP_SOURCE src/*.cc)
file(GLOB CU_SOURCE src/*.cu)
file(GLOB CPP_HEADERS inc/*.h)

#set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_FLAGS -O3)
#set_property(SOURCE ${CU_SOURCE} PROPERTY COMPILE_FLAGS -O3)
#set_property(SOURCE ${CPP_HEADERS} PROPERTY COMPILE_FLAGS -O3)



cuda_add_library(mcdnn ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE})
target_link_libraries( mcdnn
	 ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES}
)


add_executable(main main.cc)
target_link_libraries(main mcdnn ${CUDNN})


install(TARGETS mcdnn
	LIBRARY DESTINATION /usr/local/lib
	ARCHIVE DESTINATION /usr/local/lib
)
