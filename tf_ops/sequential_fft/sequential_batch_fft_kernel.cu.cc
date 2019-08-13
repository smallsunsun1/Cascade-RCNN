//
// Created by 孙嘉禾 on 2019-07-04.
//

#if GOOGLE_CUDA

#include <cufft.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
#define CUDA_MAX_NUM_THREADS 1024
#define CUDA_MAX_NUM_BLOCKS 65535
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i+= blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N) {
    int block_num = (N + CUDA_MAX_NUM_THREADS - 1) / CUDA_MAX_NUM_THREADS;
    return block_num > CUDA_MAX_NUM_BLOCKS ? CUDA_MAX_NUM_BLOCKS : block_num;
}

__global__ void complex_inplace_sacle(complex64 *data, int num, float scale) {
    // one complex 64 number is equivalent to two float number
    float *data_cast = reinterpret_cast<float *>(data);
    int num_cast = num * 2;
    CUDA_KERNEL_LOOP(i, num_cast) {
        data_cast[i] *= scale;
    }
}

__global__ void complex_inplace_scale(complex128 *data, int num, double scale) {
    // one complex128 number is equivalent to two double number
    double *data_cast = reinterpret_cast<double *>(data);
    int num_cast = num * 2;
    CUDA_KERNEL_LOOP(i, num_cast) {
        data_cast[i] *= scale;
    }
}

inline bool cufftFunc(cufftHandle plan, const complex64 *data_in,
                      complex64 *data_out, int direction) {
    // cufftExecC2C cannot take const input, hence the const_cast
    cufftComplex* in = const_cast<cufftComplex*>(reinterpret_cast<const cufftComplex*>(data_in));


}
inline bool cufftFunc(cufftHandle plan, const complex128 *data_in,
                      complex128 *data_out, int direction) {

}
}

#endif

