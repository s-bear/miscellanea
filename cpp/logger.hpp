#pragma once

// This is free and unencumbered software released into the public domain.

/* Thread-safe logging with runtime level selection and configurable streams
Uses ostringstream insertion operations to do formatting.

logging::start(logging::debug); //sets debug level and starts the logging thread
LOG(debug) << "Emit a debug message"; //Produces e.g. "2019-01-01 13:47:21 [debug] Emit a debug message"

the LOG(level) macro expands to an if statement, so disabled levels won't do any formatting

stream << logging::join(delimiter, container)
stream << logging::join(delimiter, container, format)
insert each item in container into the stream, seperated by the delimiter. The
version with format inserts format(item) into the stream.

*/

#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <atomic>

#define LOG(level) \
	if(!logging::active(logging::level)) {} \
	else logging::log_##level(__FILE__,__LINE__)

namespace logging {
    enum level_t {
        debug,
        info,
        warning,
        error,
        fatal,
		none //must be last
    };
	level_t str_to_level(const std::string& str);
	void set_level(level_t);
	level_t get_level();
	bool active(level_t);
	std::string tstamp(const std::string& fmt = "%Y-%m-%d %H:%M:%S");
	void start(level_t);
	void start();
	void stop();
	//extract level_t from any stream that supports operator>>(std::string&)
	template<typename istream_t>
	istream_t &operator>>(istream_t &in, level_t &lev) {
		std::string s;
		in >> s;
		lev = str_to_level(s);
		return in;
	}

	template<typename container_t>
	struct join_t {
		std::string delim;
		const container_t& container;
	};
	template<typename container_t, typename format_t>
	struct join_fmt_t {
		std::string delim;
		const container_t& container;
		format_t format;
	};
	template<typename string_t, typename container_t>
	join_t<container_t> join(string_t&& s, const container_t& c) {
		return join_t<container_t>{std::forward<string_t>(s), c};
	}
	template<typename string_t, typename container_t, typename format_t>
	join_fmt_t<container_t, format_t> join(string_t&& s, const container_t& c, format_t format) {
		return join_fmt_t<container_t, format_t>{std::forward<string_t>(s), c, format};
	}
	template<typename stream_t, typename container_t>
	stream_t& operator<<(stream_t &out, join_t<container_t> &j) {
		auto it = std::begin(j.container), end = std::end(j.container);
		if (it != end) out << *it; it = std::next(it);
		for (; it != end; it = std::next(it))
			out << j.delim << *it;		
		return out;
	}
	template<typename stream_t, typename container_t, typename format_t>
	stream_t& operator<<(stream_t &out, join_fmt_t<container_t, format_t> &j) {
		auto it = std::begin(j.container), end = std::end(j.container);
		if (it != end) out << j.format(*it); it = std::next(it);
		for (; it != end; it = std::next(it))
			out << j.delim << j.format(*it);
		return out;
	}

	struct logger_t;

	//thread-safe queue of std::string
	class queue_t {
		std::mutex mut;
		logger_t *logger;
		std::deque<std::string> queue;
	public:
		queue_t(logger_t*l=nullptr);
		~queue_t();
		void set_logger(logger_t*);
		void push(const std::string& item);
		void push(std::string&& item);
		bool pop(std::string& item);
	};

	//state
	struct logger_t {
		level_t _level = level_t::info;
		std::ostream* streams[level_t::none];
		queue_t queues[level_t::none];
		std::atomic_bool running;
		bool sleepy;
		std::mutex mut;
		std::condition_variable cond;
		std::thread thread;
		logger_t();
		~logger_t();
		void start();
		void stop();
		void run();
		void notify();
		bool flush();
		void set_streams(std::ostream* os);
	};
	extern logger_t logger;

    struct logline {
		queue_t &queue;
		std::ostringstream os;
        logline(queue_t &queue, const std::string& lev);
        ~logline();
        
		template<typename T>
        logline& operator<<(T&& t) {
			os << std::forward<T>(t);
            return *this;
        }
		template<typename T>
		logline& operator<<(const T * const tp) {
			if (tp) os << tp;
			else os << "nullptr";
			return *this;
		}

		typedef std::ostream& (manip)(std::ostream&);
        logline& operator<<(manip& m) {
            os << m;
            return *this;
        }
    };
#define _LOG_F(lev) \
	inline logline log_##lev(const char*, int) { \
		return logline(logger.queues[lev], #lev); \
	}

	_LOG_F(debug)
	_LOG_F(info)
	_LOG_F(warning)
	_LOG_F(error)
	_LOG_F(fatal)

#undef _LOG_F

}
