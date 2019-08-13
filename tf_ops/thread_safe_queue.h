//
// Created by 孙嘉禾 on 2019-07-16.
//

#ifndef TF_OPS_THREAD_SAFE_QUEUE_H
#define TF_OPS_THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>

template <typename T>
class thread_safe_queue{
 private:
  std::queue<T> queue;
  mutable std::mutex mut;
 public:
  bool try_pop(T& task){
      std::lock_guard<std::mutex> lock(mut);
      if (queue.empty()){
          return false;
      }
      task = std::move(queue.front());
      queue.pop();
      return true;
  }
  void push(T& task){
      std::lock_guard<std::mutex> lock(mut);
      queue.push(std::move(task));
  }
  void push(T&& task){
      std::lock_guard<std::mutex> lock(mut);
      queue.push(std::move(task));
  }
};

#endif //TF_OPS_THREAD_SAFE_QUEUE_H
