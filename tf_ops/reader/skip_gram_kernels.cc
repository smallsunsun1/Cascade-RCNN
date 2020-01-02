//
// Created by 孙嘉禾 on 2019-08-11.
//

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

REGISTER_OP("SkipGramGenerateCandidates")
    .Input("input_tensor: T")
    .Input("min_skips: int32")
    .Input("max_skips: int32")
    .Input("start: int32")
    .Input("limit: int32")
    .Input("emit_self_as_target: bool")
    .Output("tokens: T")
    .Output("labels: T")
    .Attr("T: type")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      shape_inference::ShapeHandle unused;
      // input_tensor must be of rank-1.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      // All other args must be scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));

      // Due to possible randomness in selecting skips, we only know that the
      // outputs will be of rank-1, but not their sizes.
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

template<typename T>
class SkipGramGenerateCandidatesOp : public OpKernel {
 private:
  GuardedPhiloxRandom generator_;
 public:
  explicit SkipGramGenerateCandidatesOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, generator_.Init(context));
  }
  void Compute(OpKernelContext *context) override {
      const Tensor *input_tensor;
      OP_REQUIRES_OK(context, context->input("input_tensor", &input_tensor));
      const auto input = input_tensor->flat<T>();
      const Tensor *min_skips_tensor;
      OP_REQUIRES_OK(context, context->input("min_skips", &min_skips_tensor));
      const int min_skips = *(min_skips_tensor->scalar<int>().data());
      const Tensor *max_skips_tensor;
      OP_REQUIRES_OK(context, context->input("max_skips", &max_skips_tensor));
      const int max_skips = *(max_skips_tensor->scalar<int>().data());
      const Tensor &input_check = context->input(0);
      OP_REQUIRES(context, TensorShapeUtils::IsVector(input_check.shape()),
                  errors::InvalidArgument("input_tensor must be of rank 1"));
      OP_REQUIRES(
          context, min_skips >= 0 && max_skips >= 0,
          errors::InvalidArgument("Both min_skips and max_skips must be >= 0."));
      OP_REQUIRES(context, min_skips <= max_skips,
                  errors::InvalidArgument("min_skips must be <= max_skips."));
      const Tensor *start_tensor;
      OP_REQUIRES_OK(context, context->input("start", &start_tensor));
      const int start = *(start_tensor->scalar<int>().data());
      const Tensor *limit_tensor;
      OP_REQUIRES_OK(context, context->input("limit", &limit_tensor));
      const int limit = *(limit_tensor->scalar<int>().data());
      const int end = limit < 0 ? input.size()
                                : std::min(start + limit, static_cast<int>(input.size()));
      const Tensor *emit_self_tensor;
      OP_REQUIRES_OK(context, context->input("emit_self_as_target", &emit_self_tensor));
      const bool emit_self_as_target = *(emit_self_tensor->scalar<bool>().data());
      std::vector<T> tokens;
      std::vector<T> labels;
      random::PhiloxRandom local_gen = generator_.ReserveSamples32(end - start + 1);
      random::SimplePhilox rng(&local_gen);
      for (int i = start; i < end; i++) {
          const int skips = min_skips + rng.Uniform(max_skips - min_skips + 1);
          for (int j = -skips; j <= skips; ++j) {
              if ((i + j < start) || (i + j >= end) ||
                  (j == 0 && !emit_self_as_target)) {
                  continue;
              }
              tokens.push_back(input(i));
              labels.push_back(input(i + j));

          }
      }
      Tensor *tokens_output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output("tokens", TensorShape({static_cast<int>(tokens.size())}),
                                              &tokens_output));
      Tensor *labels_output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output("labels", TensorShape({static_cast<int>(labels.size())}),
                                              &labels_output));
      OP_REQUIRES(context, tokens_output->IsSameSize(*labels_output),
                  errors::Internal(strings::StrCat(
                      "Mismatch between tokens_output shape of ",
                      tokens_output->shape().DebugString(),
                      " and labels_output shape of ",
                      labels_output->shape().DebugString(),
                      ". This should never happen - contact ami-team@ if it does.")));
      for (int i = 0; i < tokens.size(); i++) {
          tokens_output->vec<T>()(i) = tokens[i];
          labels_output->vec<T>()(i) = labels[i];
      }
  }
};

#define REGISTER_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("SkipGramGenerateCandidates") \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<type>("T"),    \
                          SkipGramGenerateCandidatesOp<type>)
REGISTER_KERNEL(string);
REGISTER_KERNEL(int64);
REGISTER_KERNEL(int32);
REGISTER_KERNEL(int16);
#undef REGISTER_KERNEL
}














