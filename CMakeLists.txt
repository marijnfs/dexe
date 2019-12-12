# CMakeLists files in this project can
# refer to the root source directory of the project as ${MCDNN_SOURCE_DIR} and

cmake_minimum_required (VERSION 3.5)
project(DEXE LANGUAGES CXX)
#SET(CMAKE_BUILD_TYPE CACHE RELEASE STRING "Build Type")

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
	message(FATAL_ERROR "In-source builds are not permitted. Make a separate folder for building:\nmkdir build; cd build; cmake ..\nBefore that, remove the files already created:\nrm -rf CMakeCache.txt CMakeFiles")
endif()

option(USE_NIFTI "Enable Nifti loading" false)
option(BUILD_SHARED_LIBS "Build a shared library" true)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(cereal REQUIRED)
find_package(CUDA QUIET REQUIRED)
find_package(CUDNN REQUIRED)

if (USE_NIFTI)
   find_library(ZNZ znz HINTS /usr/local/lib /usr/local/lib64 REQUIRED)
   find_library(niftiio niftiio HINTS /usr/local/lib /usr/local/lib64 REQUIRED)
   find_library(nifticdf nifticdf HINTS /usr/local/lib /usr/local/lib64 REQUIRED)
   add_definitions( -DUSE_NIFTI )
endif()

set(CXX_STANDARD 17)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
if(WIN32)
	set(CUDA_NVCC_FLAGS "-arch=sm_30")
else()
	set(CUDA_NVCC_FLAGS "-arch=sm_30 -std=c++14 --compiler-options -fPIC")
endif()

file(GLOB CPP_SOURCE src/*.cc)
file(GLOB CU_SOURCE src/*.cu)
file(GLOB CPP_HEADERS inc/*.h)
set(EXT_HEADERS inc/dexe.h inc/network.h inc/optimizer.h inc/tensor.h inc/util.h inc/cudavec.h inc/cudaptr.h inc/handler.h)

#set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_FLAGS -O3)
#set_property(SOURCE ${CU_SOURCE} PROPERTY COMPILE_FLAGS -O3)
#set_property(SOURCE ${CPP_HEADERS} PROPERTY COMPILE_FLAGS -O3)

set(CUDA_LINK_LIBRARIES_KEYWORD "PUBLIC")
cuda_add_library(DEXE ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE})
if(UNIX)
    target_compile_options(DEXE PRIVATE "-fPIC")
endif()
if(BUILD_SHARED_LIBS)
    target_compile_definitions(DEXE PRIVATE DEXE_DLL)
else()
    target_compile_definitions(DEXE PRIVATE DEXE_STATIC)
endif()
target_include_directories(DEXE
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<BUILD_INTERFACE:${CUDNN_INCLUDE_DIR}>
)
if (USE_NIFTI)
    target_compile_definitions(DEXE PRIVATE USE_NIFTI)
    target_include_directories(DEXE PRIVATE ${NIFTI_INCLUDE_DIR})
    target_link_libraries(DEXE PRIVATE nifticdf niftiio znz z)
endif()

target_link_libraries( DEXE
    PUBLIC 
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDNN_LIBRARIES}
        cereal
)

install(TARGETS DEXE EXPORT DEXE-targets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(FILES ${EXT_HEADERS} DESTINATION include/DEXE)

# install export target and config for find_package
install(EXPORT DEXE-targets DESTINATION lib/cmake/DEXE)
include(CMakePackageConfigHelpers)
configure_package_config_file(
	"cmake/DEXEConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/DEXEConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/DEXE"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DEXEConfig.cmake" DESTINATION "lib/cmake/DEXE")

### Build the executable
add_executable(main main.cc)
target_link_libraries(main PRIVATE DEXE)

add_executable(train train.cc)
target_link_libraries(train PRIVATE DEXE)
