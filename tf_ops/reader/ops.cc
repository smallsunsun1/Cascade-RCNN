//
// Created by 孙嘉禾 on 2019-06-19.
//

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("FormatText")
    .Output("epoch: int32")
    .Output("labels: float")
    .Output("read_features: n_real*float")
    .Output("enum_features: n_enum*int32")
    .Output("enums_ids: n_enums*int32")
    .Output("enums_vals: n_enums*int32")
    .Output("enums_weight: n_enums*float")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int batch_size;
      int n_real;
      int n_enum;
      int n_enums;
      c->GetAttr("batch_size", &batch_size);
      c->GetAttr("n_real", &n_real);
      c->GetAttr("n_enum", &n_enum);
      c->GetAttr("n_enums", &n_enums);
      auto d_batch = shape_inference::DimensionOrConstant(batch_size);
      auto d_real = shape_inference::DimensionOrConstant(n_real);
      auto d_enum = shape_inference::DimensionOrConstant(n_enum);
      auto d_1 = shape_inference::DimensionOrConstant(1);
      auto d_2 = shape_inference::DimensionOrConstant(2);
      c->set_output(0, c->MakeShape({d_1}));
      c->set_output(1, c->MakeShape({d_batch, d_1}));
      int index = 2;
      for (int i = 0; i < n_real; i++, index++) {
          c->set_output(index, c->MakeShape({d_batch}));
      }
      for (int i = 0; i < n_enum; i++, index++) {
          c->set_output(index, c->MakeShape({d_batch}));
      }
      for (int i = 0; i < n_enums; i++, index++) {
          c->set_output(index, c->MakeShape({shape_inference::InferenceContext::kUnknownDim, d_2}));
      }
      for (int i = 0; i < n_enums; i++, index++) {
          c->set_output(index, c->MakeShape({shape_inference::InferenceContext::kUnknownDim}));
      }
      for (int i = 0; i < n_enums; i++, index++) {
          c->set_output(index, c->MakeShape({shape_inference::InferenceContext::kUnknownDim}));
      }
      return Status::OK();
    })
    .SetIsStateful()
    .Attr("reader_name: string")
    .Attr("filenames: list(string)")
    .Attr("batch_size: int = 10")
    .Attr("n_real: int")
    .Attr("n_enum: int")
    .Attr("n_enums: int")
    .Attr("n_threads: int = 5")
    .Attr("qsize: int = 100")
    .Doc(R"doc(xxxx)doc");

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
}
