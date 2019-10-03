#pragma once

// This is free and unencumbered software released into the public domain.

#include <deque>
#include <mutex>
#include <condition_variable>

/* Thread-safe queue with optional maximum size 
 * queue_t must implement size(), front(), push_back(), pop_front() with the same signature and semantics as std::queue
 * When the	queue is full(), push(item) will leave the queue unchanged unless drop_oldest is true -- then it will pop_front and push_back(item) instead
*/
template<typename T, typename queue_t = std::deque<T>>
class tsqueue {
	const size_t max_size;
	const bool drop_oldest;
	std::mutex mut;
	std::condition_variable pop_cond, push_cond;
	queue_t queue;
public:
	typedef queue_t queue_type;
	typedef T value_type;
	
	//set max size, default initialize underlying queue
	tsqueue(size_t max_size = 0, bool drop_oldest = false) : max_size(max_size), drop_oldest(drop_oldest), queue() {}

	//set max size, forwards args to underlying queue
	template<typename ... Args>
	tsqueue(size_t max_size, bool drop_oldest, Args&&... args) : max_size(max_size), drop_oldest(drop_oldest), queue(std::forward<Args>(args)...) {}
	
	~tsqueue() {}
	
	// test whether the queue is empty. _empty() is not thread-safe, empty() is
	bool _empty() { //no lock
		return queue.size() == 0;
	}
	bool empty() {
		std::unique_lock lock(mut);
		return _empty();
	}
	
	// test whether the queue is full. _full() is not thread-safe, full() is
	bool _full() { //no lock
		return (max_size > 0 && queue.size() >= max_size);
	}
	bool full() {
		std::unique_lock lock(mut);
		return _full();
	}

	//push an item onto the back of the queue
	//returns false if the queue was full and an item was dropped, or if drop_oldest was set, the oldest item was dropped
	template<typename T>
	bool push(T&& item) {
		std::unique_lock lock(mut);
		if (!_full()) {
			queue.push_back(std::forward<T>(item));
			lock.unlock();
			pop_cond.notify_one();
			return true;
		}
		else if (drop_oldest) {
			queue.pop_front();
			queue.push_back(std::forward<T>(item));
			lock.unlock();
			pop_cond.notify_one();
			return false;
		}
		return false;
	}
	//push, wait for a timeout for available space
	//returns true on success
	//returns false if the timeout expired, returned spuriously, or interrupt_push() was called
	//NB this routine does not drop items from the front of the queue
	template<typename T, typename duration_t>
	bool push_wait(T&& item, const duration_t& timeout = 0) {
		std::unique_lock lock(mut);
		if (_full()) {
			push_cond.wait_for(lock, timeout);
			if (_full()) return false;
		}
		queue.push_back(std::forward<T>(item));
		lock.unlock();
		pop_cond.notify_one();
		return true;
	}
	
	//interrupts any push_wait() calls, causing them to wake immediately
	void interrupt_push() {
		push_cond.notify_all();
	}

	//pop an item from the front of the queue
	//returns true on success
	//returns false if the queue was empty (leaving item unchanged)
	bool pop(value_type& item) {
		std::unique_lock lock(mut);
		if (_empty()) return false;
		item = queue.front();
		queue.pop_front();
		lock.unlock();
		push_cond.notify_one();
		return true;
	}
	
	//pop an item from the front of the queue, wait for timeout for an item to be available
	//returns true on success
	//returns false if the timeout expired, returned spuriously, or if interrupt_pop() was called
	template<typename duration_t>
	bool pop_wait(value_type& item, const duration_t& timeout=0) {
		std::unique_lock lock(mut);
		if (_empty()) {
			pop_cond.wait_for(lock, timeout);
			if (_empty()) return false;
		}
		item = queue.front();
		queue.pop_front();
		lock.unlock();
		push_cond.notify_one();
		return true;
	}
	
	//interrupts any pop_wait() calls, causing them to wake immediately
	void interrupt_pop() {
		pop_cond.notify_all();
	}
};