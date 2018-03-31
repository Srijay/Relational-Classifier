//
// Copyright (c) 2013 Juan Palacios juan.palacios.puyana@gmail.com
// Subject to the BSD 2-Clause License
// - see < http://opensource.org/licenses/BSD-2-Clause>
//

#ifndef CONCURRENT_QUEUE_
#define CONCURRENT_QUEUE_

#include <queue>
#include <chrono>
#include <ratio>
#include <thread>
#include <mutex>
#include <condition_variable>

using std::chrono::duration;
using std::ratio;

template <typename T>
class ProducerConsumerQueue {
 public:

  bool pop(T *item, const duration<long,ratio<1,1000>> &millis) {
    std::unique_lock<std::mutex> mlock(mutex_);
    if (cond_.wait_for(mlock, millis, [this](){ return !(this->queue_.empty()); })) {
      assert(!queue_.empty());
      *item = queue_.front();
      queue_.pop();
      cond_.notify_one();
      return true;
    }
    else {
      return false;
    }
  }

  void pop(T& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty()) {
      cond_.wait(mlock);
    }
    item = queue_.front();
    queue_.pop();
    cond_.notify_one();
  }

  void push(const T& item, int waitIfLargerThan=100000) {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.size() > waitIfLargerThan) {
      cond_.wait(mlock);
    }
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }
  
  ProducerConsumerQueue()=default;
  ProducerConsumerQueue(const ProducerConsumerQueue&) = delete;            // disable copying
  ProducerConsumerQueue& operator=(const ProducerConsumerQueue&) = delete; // disable assignment
  
 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

#endif
