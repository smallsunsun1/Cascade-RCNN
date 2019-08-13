//
// Created by 孙嘉禾 on 2019-06-20.
//

#include "example.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation
template<typename T>
struct ExampleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &d, int size, const T *in, T *out) {
      for (int i = 0; i < size; ++i) {
          out[i] = 2 * in[i];
      }
  }
};

// OpKernel definition
// template parameter <T> is the datatype of the tensor.
template<typename Device, typename T>
class ExampleOp : public OpKernel {
 public:
  explicit ExampleOp(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override {
      // Grab the input tensor
      const Tensor &input_tensor = context->input(0);
      // Create an output tensor
      Tensor *output_tensor = nullptr;
      OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                  errors::InvalidArgument("Too many element in tensor"));
      ExampleFunctor<Device, T>()(context->eigen_device<Device>(),
                                  static_cast<int>(input_tensor.NumElements()),
                                  input_tensor.flat<T>().data(),
                                  output_tensor->flat<T>().data());
  }
};

#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(Name("Example").Device(DEVICE_CPU).TypeConstraint<T>("T"), ExampleOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
extern template ExampleFunctor<GPUDevice, T>; \
REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExampleOp<GPUDevice, T>);
REGISTER_GPU(float)
REGISTER_GPU(int32);
#endif
}


