//
// Created by 孙嘉禾 on 2019-06-20.
//

#ifndef TF_OPS_EXAMPLE_H
#define TF_OPS_EXAMPLE_H

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
template<typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device &d, int size, const T *in, T *out);
};

template<typename T>
struct ExampleFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice &d, int size, const T *in, T *out);
};
}

#endif //TF_OPS_EXAMPLE_H
