# CMakeLists files in this project can
# refer to the root source directory of the project as ${MCDNN_SOURCE_DIR} and

cmake_minimum_required (VERSION 2.8.11)
project (MCDNN)
#SET(CMAKE_BUILD_TYPE CACHE RELEASE STRING "Build Type")

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
	message(FATAL_ERROR "In-source builds are not permitted. Make a separate folder for building:\nmkdir build; cd build; cmake ..\nBefore that, remove the files already created:\nrm -rf CMakeCache.txt CMakeFiles")
endif()

find_package(CUDA QUIET REQUIRED)
find_package(Protobuf QUIET REQUIRED)
find_package( OpenCV REQUIRED )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS inc/img.proto inc/sample.proto)

#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-O2;-DVERBOSE")

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

set(CMAKE_CXX_FLAGS "-std=c++11")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11")
#set(CMAKE_CXX_COMPILER "nvcc")

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

cuda_add_library(mcdnn ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE} ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries( mcdnn
	-lcudnn ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} -lleveldb ${PROTOBUF_LIBRARIES} ${OpenCV_LIBS}
)

install(TARGETS mcdnn
	LIBRARY DESTINATION /usr/local/lib
	ARCHIVE DESTINATION /usr/local/lib
)
