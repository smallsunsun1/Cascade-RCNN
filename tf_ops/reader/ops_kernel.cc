//
// Created by 孙嘉禾 on 2019-06-19.
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"

#include "tensorflow/core/platform/posix/posix_file_system.h"
#include "tensorflow/core/platform/hadoop/hadoop_file_system.h"
#include "tensorflow/core/lib/gtl/stl_util.h"

#include "unsupported/Eigen/CXX11/ThreadPool"
#include "threadsafe_queue.h"

#include <thread>
#include <iostream>
#include <vector>
#include <ctime>
#include <atomic>
#include <string>
#include <memory>
#include <map>
#include <cstdio>
#include <cstdlib>

namespace tensorflow {
class Example {
 public:
  int32 epoch;
  int id;
  int batch_size;
  int n_real;
  int n_enum;
  int n_enums;
  float *a_labels;
  float **a_real_features;
  int32 **aa_enum_features;
  std::vector<int32> *enums_ids;
  std::vector<int32> *enums_val;
  std::vector<float> *enums_weight;
  TensorShape shape_labels;
  TensorShape shape_real;
  TensorShape shape_enum;
  void init(int id, int batch_size, int n_real, int n_enum, int n_enums) {
      this->id = id;
      this->batch_size = batch_size;
      this->n_real = n_real;
      this->n_enum = n_enum;
      this->n_enums = n_enums;
      a_labels = new float[batch_size];
      a_real_features = new float *[n_real];
      for (int i = 0; i < n_real; i++) {
          a_real_features[i] = new float[batch_size];
      }
      aa_enum_features = new int32 *[n_enum];
      for (int i = 0; i < n_enum; i++) {
          aa_enum_features[i] = new int32[batch_size];
      }
      enums_ids = new std::vector<int32>[n_enums];
      enums_val = new std::vector<int32>[n_enums];
      enums_weight = new std::vector<float>[n_enums];
      shape_labels = TensorShape({batch_size, 1});
      shape_real = TensorShape({batch_size});
      shape_enum = TensorShape({batch_size});
  }
  void reset() {
      for (int i = 0; i < n_enums; ++i) {
          enums_ids[i].resize(0);
          enums_val[i].resize(0);
          enums_weight[i].resize(0);
      }
  }
};

class FileReaderGen {
 private:
  std::vector<std::string> fnames;
  std::vector<int> true_index;
  int epoch = 0;
  int i = 0;
  FileSystem *p_fs;
  mutable std::mutex mtx;
 public:
  FileReaderGen(std::vector<std::string> fnames) {
      if (fnames.size() == 0) {
          printf("size of filenames must > 0 \n");
          return;
      }
      if (str_util::StartsWith(StringPiece(fnames[0]), StringPiece("hdfs://"))) {
          printf("read hdfs\n");
          p_fs = new HadoopFileSystem();
      } else {
          printf("read local\n");
          p_fs = new PosixFileSystem();
      }
      this->fnames = fnames;
  }
  int next(std::unique_ptr<RandomAccessFile> *p_reader) {
      int ret = epoch;
      mtx.lock();
      while (true) {
          if (epoch == 0) {
              const std::string &fname = fnames[i];
              p_fs->NewRandomAccessFile(fname, p_reader);
              if (*p_reader) {
                  true_index.emplace_back(i);
                  ret = epoch;
              }
              if (++i == fnames.size()) {
                  i = 0;
                  ++epoch;
              }
              if (*p_reader)
                  break;
          } else {
              if (true_index.size() == 0) {
                  printf("true_index_size error \n");
                  ret = -1;
                  break;
              } else {
                  p_fs->NewRandomAccessFile(fnames[true_index[i]], p_reader);
                  if (*p_reader) {
                      ret = epoch;
                  }
                  if (++i == true_index.size()) {
                      i = 0;
                      ++epoch;
                      if (epoch - ret > 1) {
                          ret = -1;
                          printf("出现错误 \n");
                          break;
                      }
                  }
                  if (*p_reader)
                      break;
              }
          }
      }
      mtx.unlock();
      return ret;
  }
};

struct ThreadArgs {
  std::atomic<bool> *p_running;
  std::atomic<int> *p_n_running_threads;
  int batch_size;
  int n_real;
  int n_enum;
  int n_enums;
  threadsafe_queue<Example *> *q_feed;
  threadsafe_queue<Example *> *q_fetch;
  FileReaderGen *fgen;
};

inline bool _tof(float &value, char *&bufp, char *buf, char *content, char *&p, char *&begin) {
    bool ret = true;
    if (bufp > buf) {
        memcpy(bufp, content, (p - content));
        *(bufp + (p - content)) = '\0';
        value = atof(buf);
        bufp = buf;
    } else if (p > begin) {
        *p = '\0';
        value = atof(begin);
    } else {
        ret = false;
    }
    begin = p + 1;
    return ret;
}

inline bool _toi(int32 &value, char *&bufp, char *buf, char *content, char *&p, char *&begin) {
    bool ret = true;
    if (bufp > buf) {
        memcpy(bufp, content, (p - content));
        *(bufp + (p - content)) = '\0';
        value = atoi(buf);
        bufp = buf;
    } else if (p > begin) {
        *p = '\0';
        value = atoi(begin);
    } else {
        ret = false;
    }
    begin = p + 1;
    return ret;
}

void *feed_queue_node(void *args) {
    ThreadArgs *p_args = (ThreadArgs *) args;
    FileReaderGen *fgen = p_args->fgen;
    std::unique_ptr<RandomAccessFile> reader;
    int epoch = fgen->next(&reader);
    if (-1 == epoch) {
        printf("Some error happened! \n");
        (*(p_args->p_n_running_threads)) -= 1;
        return nullptr;
    }
    int container_size = 2000000;
    char *content = new char[container_size + 1];
    StringPiece result;
    int s = -1;
    int buf_size = 200000;
    char *buf = new char[buf_size];
    char *bufp = buf;
    char *begin;
    int i;
    int j;
    Example *p_example = nullptr;
    uint64 offset = 0;
    int line_num = 0;
    int batch_size = p_args->batch_size;
    clock_t t0, t1, t2;
    bool has_weight = false;
    std::atomic<bool> &running = *(p_args->p_running);
    while (running) {
        reader->Read(offset, container_size, &result, content);
        int result_size = result.size();
        if (result_size < container_size) {
            if (result_size == 0 || content[result_size - 1] != '\n') {
                content[result_size] = '\n';
                result_size += 1;
            }
        }
        offset += result.size();
        if (result_size > 0) {
            begin = content;
            for (char *p = content, *n = content + result_size; p < n; ++p) {
                if (s == 2) {
                    if (*p == ',' || *p == '\t') {
                        bool part_end = (*p == '\t');
                        float value;
                        if (_tof(value, bufp, buf, content, p, begin) && i < p_example->n_real) {
                            if (p_example) {
                                p_example->a_real_features[i++][line_num] = value;
                            }
                        }
                        if (part_end) {
                            s = 3;
                            i = 0;
                        }
                    }
                }else if (s == 3){
                    // enum_type_features
                }
            }
        }
    }
}
}
