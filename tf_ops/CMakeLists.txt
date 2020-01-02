cmake_minimum_required(VERSION 3.14)
project(tf_ops)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_SHARED_LIBRARY_SUFFIX .so)
set(CMAKE_SHARED_LINKER_FLAGS "-fPIC")
set(CMAKE_BUILD_TYPE Release)

add_definitions("-DGOOGLE_CUDA")
#set(CMAKE_CXX_COMPILER_ABI 0)
#set(CMAKE_VERBOSE_MAKEFILE ON)
#target_link_libraries(ops PRIVATE $ENV{TF_LIB_PATH}/python/_pywrap_tensorflow_internal.so)

set(ENV{TF_INCLUDE_PATH} /Users/sunjiahe/miniconda3/lib/python3.6/site-packages/tensorflow/include)
add_library(ops SHARED reader/skip_gram_kernels.cc reader/to_zeros.cc)
target_include_directories(ops PRIVATE $ENV{TF_INCLUDE_PATH})
#target_include_directories(ops PUBLIC /usr/local/Cellar/eigen/3.3.7/include/eigen3)
target_link_libraries(ops PRIVATE /Users/sunjiahe/miniconda3/lib/python3.6/site-packages/tensorflow/libtensorflow_framework.so)
target_link_options(ops PRIVATE -undefined dynamic_lookup)
#target_compile_options(ops PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
#target_compile_definitions(ops PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)

add_subdirectory(solutions)
#file(GLOB mkl_libs /opt/intel/mkl/lib/*)
add_executable(a main.cpp thread_safe_queue.cc thread_safe_queue.h)
target_include_directories(a PRIVATE $ENV{TF_INCLUDE_PATH})
target_compile_options(a PRIVATE -march=native)

#add_subdirectory(sequential_fft)