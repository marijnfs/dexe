# CMakeLists files in this project can
# refer to the root source directory of the project as ${MCDNN_SOURCE_DIR} and

cmake_minimum_required (VERSION 2.8.11)
project (MCDNN)

find_package(CUDA QUIET REQUIRED)

#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-O2;-DVERBOSE")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "-std=c++11;-O3")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
set(CMAKE_CXX_COMPILER "nvcc")

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
	-lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options
)

install(TARGETS mcdnn
	LIBRARY DESTINATION /usr/local/lib
	ARCHIVE DESTINATION /usr/local/lib
)