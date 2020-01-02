//
// Created by 孙嘉禾 on 2019-08-12.
//

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
REGISTER_OP("ToZeros")
    .Attr("T: {float, int32, double}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return tensorflow::Status::OK();
    });

template <typename Device, typename T>
struct ToZerosForward{
  void operator()(const Device& d,
      typename TTypes<T, 1>::Flat output){
      output.setConstant(0);
  }
};

template <typename Device, typename T>
class ToZerosOp: public tensorflow::OpKernel{
 public:
  explicit ToZerosOp(tensorflow::OpKernelConstruction* ctx):OpKernel(ctx){}
  void Compute(OpKernelContext* ctx) override {
      const Tensor& input = ctx->input(0);
      TensorShape out_shape = input.shape();
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
      auto output_flat = output->flat<T>();
      ToZerosForward<Device, T>()(ctx->eigen_device<Device>(), output_flat);
  }
};

#define REGISTER(T) \
REGISTER_KERNEL_BUILDER(Name("ToZeros").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    ToZerosOp<Eigen::ThreadPoolDevice, T>);

REGISTER(int32)
REGISTER(double)
REGISTER(float)


#undef REGISTER

}
