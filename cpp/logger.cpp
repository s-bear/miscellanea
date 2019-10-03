
// This is free and unencumbered software released into the public domain.

#include "logger.hpp"
#include <sstream>
#include <cstdio>
#include <functional>

namespace logging {
	
    logger_t logger;

	level_t str_to_level(const std::string& str) {
        if(str == "debug") return logging::debug;
        else if(str == "info") return logging::info;
        else if(str == "warning") return logging::warning;
        else if(str == "error") return logging::error;
        else if(str == "fatal") return logging::fatal;
        else if(str == "none") return logging::none;
        else throw std::runtime_error("Invalid logging::level: " + str);
    }

    void set_level(level_t lev) {
        logger._level = lev;
    }
	level_t get_level() {
		return logger._level;
	}
	bool active(level_t lev) {
		return logger._level <= lev;
	}

	std::string tstamp(const std::string& fmt) {
		time_t now = time(nullptr);
		tm gmtnow;
#ifdef WIN32
		gmtime_s(&gmtnow, &now);
#else
		gmtnow = *gmtime(&now);
#endif
		char buffer[128];
		strftime(buffer, 128, fmt.c_str(), &gmtnow);
		return std::string(buffer);
	}

	void start(level_t lev) {
		logger._level = lev;
		logger.start();
	}

	void start() {
		logger.start();
	}

	void stop() {
		logger.stop();
	}

	queue_t::queue_t(logger_t *logger) : logger(logger), mut(), queue() {}
	queue_t::~queue_t() {}
	void queue_t::push(const std::string& item) {
		std::unique_lock lock(mut);
		queue.push_back(item);
		if (logger) {
			lock.unlock();
			logger->notify();
		}
	}

	void queue_t::push(std::string&& item) {
		std::unique_lock lock(mut);
		queue.push_back(std::move(item));
		if (logger) {
			lock.unlock();
			logger->notify();
		}
	}

	bool queue_t::pop(std::string& item) {
		std::unique_lock lock(mut);
		if (queue.empty()) return false;
		item = queue.front();
		queue.pop_front();
		return true;
	}

	void queue_t::set_logger(logger_t* logger) {
		this->logger = logger;
	}

	logger_t::logger_t() : _level(info), running(false), sleepy(false) {
		streams[debug] = &std::cout;
		streams[info] = &std::cout;
		streams[warning] = &std::cout;
		streams[error] = &std::cerr;
		streams[fatal] = &std::cerr;
		for (size_t i = 0; i < level_t::none; ++i)
			queues[i].set_logger(this);
	}

    logger_t::~logger_t() {
		stop();
		flush();
		log_info(__FILE__,__LINE__) << "END LOG";
		flush();
    }

	void logger_t::start() {
		if (!running.exchange(true)) {
			thread = std::thread(std::mem_fn(&logger_t::run), &logger);
		}
	}

	void logger_t::stop() {
		if (running.exchange(false) && thread.joinable()) {
			notify();
			thread.join();
		}
	}

	void logger_t::run() {
		while (running) {
			if (!flush()) {
				std::unique_lock lock(mut);
				sleepy = true;
				while(sleepy) cond.wait_for(lock, std::chrono::milliseconds(200));
			}
		}
	}

	void logger_t::notify() {
		std::unique_lock lock(mut);
		if (sleepy) {
			sleepy = false;
			cond.notify_one();
		}
	}

	bool logger_t::flush() {
		std::string s;
		bool ret = false;
		for (int i = 0; i < level_t::none; ++i) {
			if (streams[i] != nullptr) {
				while (queues[i].pop(s)) {
					ret = true;
					(*streams[i]) << s << std::endl;
				}
			}
		}
		return ret;
	}

	void logger_t::set_streams(std::ostream* os) {
		for (int i = 0; i < level_t::none; ++i)
			streams[i] = os;
	}

    logline::logline(queue_t& queue, const std::string& level) : queue(queue) {
        os << tstamp() << " [" << level << "] ";
    }

    logline::~logline()  {
		queue.push(os.str());
    }
}

