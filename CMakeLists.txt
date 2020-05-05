# CMakeLists files in this project can
# refer to the root source directory of the project as ${MCDNN_SOURCE_DIR} and

cmake_minimum_required (VERSION 3.5)
project(dexe LANGUAGES CXX)
#SET(CMAKE_BUILD_TYPE CACHE RELEASE STRING "Build Type")

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
	message(FATAL_ERROR "In-source builds are not permitted. Make a separate folder for building:\nmkdir build; cd build; cmake ..\nBefore that, remove the files already created:\nrm -rf CMakeCache.txt CMakeFiles")
endif()

option(USE_NIFTI "Enable Nifti loading" false)
option(BUILD_SHARED_LIBS "Build a shared library" true)

set(CMAKE_DEBUG_POSTFIX "d")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(CUDA QUIET REQUIRED)
find_package(CUDNN REQUIRED)

CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_NVCC_FLAGS ${CUDA_ARCH_BIN})

if (USE_NIFTI)
   find_library(ZNZ znz HINTS /usr/local/lib /usr/local/lib64 REQUIRED)
   find_library(niftiio niftiio HINTS /usr/local/lib /usr/local/lib64 REQUIRED)
   find_library(nifticdf nifticdf HINTS /usr/local/lib /usr/local/lib64 REQUIRED)
   add_definitions( -DUSE_NIFTI )
endif()

set(CXX_STANDARD 17)
set(CUDA_PROPAGATE_HOST_FLAGS ON)

file(GLOB CPP_SOURCE src/*.cc)
file(GLOB CU_SOURCE src/*.cu)
file(GLOB CPP_HEADERS inc/dexe/*.h)
set(EXT_HEADERS inc/dexe/dexe.h inc/dexe/models.h inc/dexe/network.h inc/dexe/optimizer.h inc/dexe/tensor.h inc/dexe/util.h inc/dexe/cudavec.h inc/dexe/handler.h inc/dexe/config.h inc/dexe/print.h inc/dexe/io.h)

#set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_FLAGS -O3)
#set_property(SOURCE ${CU_SOURCE} PROPERTY COMPILE_FLAGS -O3)
#set_property(SOURCE ${CPP_HEADERS} PROPERTY COMPILE_FLAGS -O3)

set(CUDA_LINK_LIBRARIES_KEYWORD "PUBLIC")
cuda_add_library(dexe ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE})
if(UNIX)
    target_compile_options(dexe PRIVATE "-fPIC")
endif()
if(BUILD_SHARED_LIBS)
    target_compile_definitions(dexe PRIVATE DEXE_DLL)
else()
    target_compile_definitions(dexe PRIVATE DEXE_STATIC)
endif()
target_include_directories(dexe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
        $<BUILD_INTERFACE:${CUDNN_INCLUDE_DIR}>
		$<INSTALL_INTERFACE:include>
)

if (USE_NIFTI)
    target_compile_definitions(dexe PUBLIC USE_NIFTI)
    target_include_directories(dexe PUBLIC ${NIFTI_INCLUDE_DIR})
    target_link_libraries(dexe PUBLIC nifticdf niftiio znz z)
endif()

target_link_libraries( dexe
    PUBLIC 
        $<BUILD_INTERFACE:${CUDA_curand_LIBRARY}>
        $<BUILD_INTERFACE:${CUDA_CUBLAS_LIBRARIES}>
        $<BUILD_INTERFACE:${CUDNN_LIBRARIES}>
)

install(TARGETS dexe EXPORT dexe-targets RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(FILES ${EXT_HEADERS} DESTINATION include/dexe)

# install export target and config for find_package
install(EXPORT dexe-targets DESTINATION lib/cmake)
include(CMakePackageConfigHelpers)
configure_package_config_file(
	"cmake/dexeConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/dexeConfig.cmake"
	INSTALL_DESTINATION "lib/cmake"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dexeConfig.cmake" DESTINATION "lib/cmake")

option(BUILD_BINARIES "enable building binaries" OFF)

if (BUILD_BINARIES)
### Build the executable
add_executable(main bin/main.cc)
target_link_libraries(main PRIVATE dexe)

add_executable(train bin/train.cc)
target_link_libraries(train PRIVATE dexe)
endif()
