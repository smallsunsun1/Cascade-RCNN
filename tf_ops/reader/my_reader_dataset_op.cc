//
// Created by 孙嘉禾 on 2019-08-09.
//

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "fstream"

using tensorflow::DT_STRING;
using tensorflow::PartialTensorShape;
using tensorflow::Status;

class MyReaderDatasetOp : public tensorflow::data::DatasetOpKernel {
 public:
  explicit MyReaderDatasetOp(tensorflow::OpKernelConstruction *ctx) : DatasetOpKernel(ctx) {
      // Parse and validate any attrs that define the dataset using
      // `ctx->GetAttr()`, and store them in member variables.
  }
  void MakeDataset(tensorflow::OpKernelContext *ctx,
                   tensorflow::data::DatasetBase **output) override {
      // Parse and validate any input tensors that define the dataset using
      // `ctx->input()` or the utility function
      // `ParseScalarArgument<T>(ctx, &arg)`.

      // Create the dataset object, passing any (already-validated) arguments from
      // attrs or input tensors.
      *output = new Dataset(ctx);
  }
 private:
  class Dataset : public tensorflow::DatasetBase {
   public:
    Dataset(tensorflow::OpKernelContext *ctx) : tensorflow::data::DatasetBase(tensorflow::DatasetContext(ctx)) {}
    std::unique_ptr<tensorflow::IteratorBase> MakeIteratorInternal(const std::string &prefix) const override {
        return std::unique_ptr<tensorflow::IteratorBase>(new Iterator(
            {this, tensorflow::strings::StrCat(prefix, "::MyReader")}
        ));
    }
    // Record structure: Each record is represented by a scalar string tensor.
    //
    // Dataset elements can have a fixed number of components of different
    // types and shapes; replace the following two methods to customize this
    // aspect of the dataset.
    const tensorflow::DataTypeVector &output_dtypes() const override {
        static auto *const dtypes = new tensorflow::DataTypeVector({DT_STRING});
        return *dtypes;
    }
    const std::vector<PartialTensorShape> &output_shapes() const override {
        static std::vector<PartialTensorShape> *shapes = new std::vector<PartialTensorShape>({{}});
        return *shapes;
    }
    std::string DebugString() const override { return "MyReaderDatasetOp::Dataset"; }
   protected:
    // Optional: Implementation of `GraphDef` serialization for this dataset.
    //
    // Implement this method if you want to be able to save and restore
    // instances of this dataset (and any iterators over it).
    Status AsGraphDefInternal(tensorflow::SerializationContext *ctx,
                              DatasetGraphDefBuilder *b, tensorflow::Node **output) const override {
        // Construct nodes to represent any of the input tensors from this
        // object's member variables using `b->AddScalar()` and `b->AddVector()`.
        std::vector<tensorflow::Node *> input_tensors;
        TF_RETURN_IF_ERROR(b->AddDataset(this, input_tensors, output));
        return Status::OK();
    }
   private:
    class Iterator : public tensorflow::DatasetIterator<Dataset> {
     private:
      tensorflow::mutex mu_;
      tensorflow::int64 i_ GUARDED_BY(mu_);
      std::fstream fs;
     public:
      explicit Iterator(const Params &params) : DatasetIterator<Dataset>(params), i_(0) {

      }
      // Implementation of the reading logic.
      //
      // The example implementation in this file yields the string "MyReader!"
      // ten times. In general there are three cases:
      //
      // 1. If an element is successfully read, store it as one or more tensors
      //    in `*out_tensors`, set `*end_of_sequence = false` and return
      //    `Status::OK()`.
      // 2. If the end of input is reached, set `*end_of_sequence = true` and
      //    return `Status::OK()`.
      // 3. If an error occurs, return an error status using one of the helper
      //    functions from "tensorflow/core/lib/core/errors.h".
      Status GetNextInternal(tensorflow::IteratorContext *ctx,
                             std::vector<tensorflow::Tensor> *out_tensors,
                             bool *end_of_sequence) override {
          // NOTE: `GetNextInternal()` may be called concurrently, so it is
          // recommended that you protect the iterator state with a mutex.
          tensorflow::mutex_lock l(mu_);

          if (i_ < 10) {
              tensorflow::Tensor record_tensor(ctx->allocator({}), DT_STRING, {});
              record_tensor.scalar<std::string>()() = "MyReader!";
              out_tensors->emplace_back(std::move(record_tensor));
              ++i_;
              *end_of_sequence = false;
          } else {
              *end_of_sequence = true;
          }
          return Status::OK();
      }
     protected:
      // Optional: Implementation of iterator state serialization for this
      // iterator.
      //
      // Implement these two methods if you want to be able to save and restore
      // instances of this iterator.
      Status SaveInternal(tensorflow::IteratorStateWriter *writer) override {
          tensorflow::mutex_lock l(mu_);
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("i"), i_));
          return Status::OK();
      }
      Status RestoreInternal(tensorflow::IteratorContext *ctx,
                             tensorflow::IteratorStateReader *reader) override {
          tensorflow::mutex_lock l(mu_);
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("i"), &i_));
          return Status::OK();
      }
    };
  };
};

namespace tensorflow {
class TextLineReader : public ReaderBase {
 private:
  enum { kBufferSize = 256 << 10 };
  const int skip_header_lines_;
  Env *const env;
  int64 line_number_;
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::InputBuffer> input_buffer_;
 public:
  TextLineReader(const string &node_name, int skip_header_lines, Env *env)
      : ReaderBase(strings::StrCat("TextLineReader '", node_name, "'")),
        skip_header_lines_(skip_header_lines),
        env(env),
        line_number_(0) {}
  Status OnWorkStartedLocked() override {
      line_number_ = 0;
      TF_RETURN_IF_ERROR(env->NewRandomAccessFile(current_work(), &file_));
      input_buffer_.reset(new io::InputBuffer(file_.get(), kBufferSize));
      for (; line_number_ < skip_header_lines_; ++line_number_) {
          string line_contents;
          Status status = input_buffer_->ReadLine(&line_contents);
          if (errors::IsOutOfRange(status)) {
              return Status::OK();
          }
          TF_RETURN_IF_ERROR(status);
      }
      return Status::OK();
  }
  Status OnWorkFinishedLocked() override {
      input_buffer_.reset(nullptr);
      return Status::OK();
  }
  Status ReadLocked(string *key, string *value, bool *produced, bool *at_end) override {
      Status status = input_buffer_->ReadLine(value);
      ++line_number_;
      if (status.ok()) {
          *key = strings::StrCat(current_work(), ":", line_number_);
          *produced = true;
          return status;
      }
      if (errors::IsOutOfRange(status)) {
          *at_end = true;
          return Status::OK();
      } else {
          return status;
      }
  }
  Status ResetLocked() override {
      line_number_ = 0;
      input_buffer_.reset(nullptr);
      return ReaderBase::ResetLocked();
  }
};

class TextLineReaderOp : public ReaderOpKernel {
 public:
  explicit TextLineReaderOp(OpKernelConstruction *context) :
      ReaderOpKernel(context) {
      int skip_header_lines = -1;
      OP_REQUIRES_OK(context, context->GetAttr("skip_header_lines", &skip_header_lines));
      OP_REQUIRES(context, skip_header_lines >= 0,
                  errors::InvalidArgument("skip_header_lines must be >= 0 not", skip_header_lines));
      Env* env = context->env();
      SetReaderFactory([this, skip_header_lines, env](){
        return new TextLineReader(name(), skip_header_lines, env);
      });
  }
};

}

REGISTER_OP("MyReaderDataset")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MyReaderDataset").Device(tensorflow::DEVICE_CPU),
                        MyReaderDatasetOp);

REGISTER_OP("TextLineReader")
.Output("handle: variant")
.SetIsStateful()
.SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("TextLineReader").Device(tensorflow::DEVICE_CPU),
    tensorflow::TextLineReaderOp);
















