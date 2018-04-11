#ifndef _NCC_THREADPOOL_H_
#define _NCC_THREADPOOL_H_ 1

/**
 * https://alexagafonov.com/2015/05/05/thread-pool-implementation-in-c-11/
 */

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <unistd.h>

namespace ncc {
using namespace std;

class ThreadPool {
public:
  ThreadPool(int threads) :
    terminate(false),
    stopped(false) {
    // Create number of required threads and add them to the thread pool vector.
    for(int i = 0; i < threads; i++) {
      threadPool.emplace_back(thread(&ThreadPool::Invoke, this));
    }
  }

  ~ThreadPool() {
    if (!stopped) {
      ShutDown();
    }
  }

  // Adds task to a task queue.
  void Enqueue(function<void(void)> f) {
    // Scope based locking.
    {
      // Put unique lock on task mutex.
      unique_lock<mutex> lock(tasksMutex);
      // Push task into queue.
      tasks.push(f);
    }
    // Wake up one thread.
    condition.notify_one();
  }

  // Shut down the pool.
  void ShutDown() {
    // Scope based locking.
    {
      // Put unique lock on task mutex.
      unique_lock<mutex> lock(tasksMutex);
      // Set termination flag to true.
      terminate = true;
    }
    // Wake up all threads.
    condition.notify_all();
    // Join all threads.
    for(thread &thread : threadPool){
      thread.join();
    }
    // Empty workers vector.
    threadPool.empty();
    // Indicate that the pool has been shut down.
    stopped = true;
  }

private:
  // Thread pool storage.
  vector<thread> threadPool;

  // Queue to keep track of incoming tasks.
  queue<function<void(void)>> tasks;

  // Task queue mutex.
  mutex tasksMutex;

  // Condition variable.
  condition_variable condition;

  // Indicates that pool needs to be shut down.
  bool terminate;

  // Indicates that pool has been terminated.
  bool stopped;

  // Function that will be invoked by our threads.
  void Invoke() {
    function<void()> task;
    while(true) {
      // Scope based locking.
      {
	// Put unique lock on task mutex.
	unique_lock<mutex> lock(tasksMutex);
	// Wait until queue is not empty or termination signal is sent.
	condition.wait(lock, [this]{ return !tasks.empty() || terminate; });
	// If termination signal received and queue is empty
	// then exit else continue clearing the queue.
	if (terminate && tasks.empty()) {
	  return;
	}
	// Get next task in the queue.
	task = tasks.front();
	// Remove it from the queue.
	tasks.pop();
      }
      // Execute the task.
      task();
    }
  }
  
};

}  // namespace ncc

/*

Test harness:

#include "threadpool.h"

int main() {
  ncc::ThreadPool pool(5);
  for (int i = 0; i < 20; i++) {
    std::cout << "Enquing: " << i << std::endl;
    pool.Enqueue([i](){
	int sleep = (rand() % 1000000) + 1000000;
	usleep(sleep);
	std::cout << "Processed: " << i << " Sleep Time: " << sleep << std::endl;
      });
  }
  return 0;
}

 */

#endif // _NCC_THREADPOOL_H_
