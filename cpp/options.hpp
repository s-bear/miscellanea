#pragma once

// This is free and unencumbered software released into the public domain.

/* basic options parsing

Still needs a bit of work!

*/

#include <functional>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <any>
#include <string>

#include <cctype>
#include <vector>
#include <unordered_map>
#include <memory>
#include <typeinfo>

#include "iniparser.h"

namespace options {
	// anonymous namespace for helpers - not visible outside this header file
	namespace { 
		template<typename items_t>
		std::string _join(const std::string &delim, const items_t &items, const std::string& pfx = {}) {
			std::ostringstream oss;
			auto it = std::begin(items);
			oss << pfx << *it;
			it = std::next(it);
			for (; it != std::end(items); it = std::next(it))
				oss << delim << pfx << *it;
			return oss.str();
		}

		template<typename item_t>
		std::string _str(const item_t& item) {
			std::ostringstream oss;
			oss.setf(std::ios::boolalpha);
			oss << item;
			return oss.str();
		}

		template<typename str_t>
		str_t _lower(const str_t &str) {
			str_t lower;
			lower.reserve(str.size());
			for (auto c : str) lower.push_back(tolower(c));
			return lower;
		}

		//store a value to a reference using =
		template<typename T>
		struct implicit_store {
			T &storage, value;
			implicit_store(T &storage, const T &value) : storage(storage), value(value) {}
			implicit_store(T &storage, T &&value) : storage(storage), value(std::move(value)) {}
			void operator()() {
				storage = value;
			}
		};

		//extract an item from an istringstream and store it using a reference
		template<typename T>
		struct extract_store {
			T &storage;
			extract_store(T &storage) : storage(storage) {}
			void operator()(const std::string& s) {
				std::istringstream iss(s);
				iss.setf(std::ios::boolalpha);
				iss >> storage;
			}
		};

		//option with types erased behind std::function
		struct _option {
			std::vector<std::string> names;
			//help strings
			std::string help, help_type, help_implicit;
			bool parsed = false;
			std::function<void(const std::string&)> action;
			std::function<void(void)> implicit;

			_option() = default;

			template<typename ...Args>
			explicit _option(std::string tname, std::function<void(const std::string&)> action,
				Args&& ...args) :
				help_type(std::move(tname)),
				action(std::move(action)),
				names({ std::forward<Args>(args)... }) {}
		};

		typedef std::shared_ptr<_option> _option_p;
	} //end anon namespace

	//access an option in a type-safe way
	template<typename T>
	class option {
		friend class option_set;
	protected:
		_option &opt;
		T &storage;
		option(_option &opt, T &storage) : opt(opt), storage(storage) {}
	public:
		option& help(std::string help_string) {
			opt.help = std::move(help_string);
			return *this;
		}
		option& implicit(const T &value) {
			opt.implicit = implicit_store<T>(storage, value);
			opt.help_implicit = _str(value);
			return *this;
		}
		option& implicit(T &&value) {
			opt.implicit = implicit_store<T>(storage, std::move(value));
			opt.help_implicit = _str(value);
			return *this;
		}
		option& action(const std::function<void(const std::string&)> &act) {
			opt.action = act;
			return *this;
		}
		option& action(std::function<void(const std::string&)> &&act) {
			opt.action = std::move(act);
			return *this;
		}
	};

	class option_set {
		friend class parser;
		std::string _name;
		std::vector<_option_p> _options;
		std::unordered_map<std::string, _option_p> _option_map;

		_option_p _get(const std::string& name) const {
			auto it = _option_map.find(name);
			if (it != _option_map.end()) {
				return it->second;
			}
			else {
				return _option_p(nullptr);
			}
		}
	public:
		explicit option_set(std::string name = {}) : _name(std::move(name)) {}

		//check to see if an option was parsed
		bool operator[](const std::string &name) {
			auto arg = _get(name);
			if (arg) return arg->parsed;
			else return false;
		}

		template<typename T, typename ...Args>
		option<T> add(T& storage, Args&& ...names) {
			//construct the _option
			_option_p new_arg = std::make_shared<_option>(typeid(T).name(), extract_store<T>(storage), std::forward<Args>(names)...);
			//check the names
			for (const auto &name : new_arg->names) {
				auto arg = _get(name);
				if (arg) {
					throw std::runtime_error("Option name conflict: " + _join(", ", new_arg->names) + " with " + _join(", ", arg->names));
				}
			}
			//OK, store
			_options.push_back(new_arg);
			for (const auto &name : new_arg->names)
				_option_map.insert({ name, new_arg });
			//wrap in option<T> and return
			return option<T>(*new_arg, storage);
		}

		template<typename ostream_t>
		ostream_t& print(ostream_t& out) {
			if (!_name.empty())
				out << "[" << _name << "]";
			for (const auto& opt : _options) {
				out << "\n  ";
				if (_name.empty())
					out << _join(" | ", opt->names);
				else
					out << _join(" | ", opt->names, _name + ".");
				out << " : " << opt->help_type;
				if (!opt->help_implicit.empty())
					out << "(" << opt->help_implicit << ")";
				if (!opt->help.empty()) out << "\n    " << opt->help;
			}
		}
	};

	class parser {
		std::string _program_name;
		option_set _default_set;
		std::unordered_map<std::string, std::reference_wrapper<option_set>> _option_sets;

		option_set &_get(const std::string &name) {
			auto it = _option_sets.find(name);
			if (it == _option_sets.end()) {
				if (name.empty()) {
					_option_sets.insert({ name, std::ref(_default_set) });
					return _default_set;
				}
				else {
					throw std::runtime_error("No such option_set: \"" + name + "\"");
				}
			}
			return it->second.get();
		}

		void _set(const ini::item& item, bool overwrite) {
			option_set &set = _get(item.section);
			auto opt = set._get(item.key);
			if (!opt) throw std::runtime_error("Unknown option: " + item.section + "." + item.key);
			else if (overwrite || !opt->parsed) {
				if (opt->implicit) opt->implicit();	
				if (opt->action && !item.value.empty()) opt->action(item.value);
			}
		}

	public:
		template<typename T, typename ...Args>
		option<T> add(T& storage, Args&& ...names) {
			return _get({}).add(storage, std::forward<Args>(names)...);
		}
		void add(option_set &opts) {
			if (_option_sets.find(opts._name) != _option_sets.end())
				throw std::runtime_error("option_set already added: " + opts._name);
			_option_sets.insert({ opts._name, std::ref(opts) });
		}
		option_set &operator[](const std::string& name) {
			return _get(name);
		}
	
		explicit parser(std::string program_name = {}) :
			_program_name(std::move(program_name))
		{}

		void parse_args(int argc, const char * const argv[], bool overwrite = false) {
			//argv[0] is the program name
			if (_program_name.empty() && argc > 0)
				_program_name = argv[0];
			
			ini::item arg;
			for (int i = 1; i < argc; ++i) {	
				std::string token = argv[i];
				size_t dot = token.find('.');
				size_t eq = token.find('=');
				if (dot > eq) dot = std::string::npos;
				if (dot == std::string::npos) {
					//no dot
					arg.section = {};
					arg.key = token.substr(0, eq);
				}
				else {
					if (dot > 0) //keep same section as previous if . is first
						arg.section = token.substr(0, dot);
					if (dot + 1 < token.size())
						arg.key = token.substr(dot + 1, eq);
					else
						arg.key = {};
				}
				if(eq != std::string::npos && eq+1 < token.size())
					arg.value = token.substr(eq + 1);
				_set(arg, overwrite);
			}
		}

		void load(const std::string &inifile, bool overwrite = false) {
			auto params = ini::load(inifile);
			for (const auto &p : params) {
				_set(p, overwrite);
			}
		}

		template<typename ostream_t>
		ostream_t& print_help(ostream_t &out) {
			out << std::left;
			out << "Usage: " << _program_name << " [options] ";
			for (const auto& name_set : _option_sets) {
				const auto &set = name_set.second.get();
				
			}
			return out;
		}
	};
	
	template<typename ostream_t>
	ostream_t& operator<<(ostream_t& out, const parser& p) {
		return p.print_help(out);
	}
}