#include "iniparser.h"
#include <cctype>

std::istream& ini::getline_s(std::istream &is, std::string &s) {
	s.clear();
	std::istream::sentry se(is, true);
	std::streambuf *sb = is.rdbuf();
	for (;;) {
		int c = sb->sbumpc();
		switch (c) {
		case '\n': return is;
		case '\r':
			if (sb->sgetc() == '\n') sb->sbumpc();
			return is;
		case std::streambuf::traits_type::eof():
			if (s.empty()) is.setstate(std::ios::eofbit); //no line ending on last line
			return is;
		default:
			s += (char)c;
		}
	}
}

std::vector<ini::item> ini::load(const std::string &fname, char comment, char delim) {
	std::ifstream ifs(fname);
	if (!ifs) throw std::runtime_error("ini::load(): unable to open file: " + fname);
	return load(ifs, comment, delim);
}

std::vector<ini::item> ini::load(std::istream &in, char comment, char delim) {
	std::string line;
	std::string section;
	std::vector<item> items;
	for (int line_num = 0; !getline_s(in, line).eof(); ++line_num) {
		item i;
		if (line.empty()) continue;
		auto it = line.begin();
		//ignore leading whitespace
		for (; it != line.end() && std::isspace(*it); ++it);
		//section header?
		if (it != line.end() && *it == '[') {
			section.clear();
			++it;
			//append to section name until we hit ] or ;
			for (; it != line.end() && *it != ']' && *it != comment; ++it) section += *it;
		}
		else {
			//append to key until we hit the end, =, or ;
			//ignore trailing whitespace
			std::string ws;
			for (; it != line.end() && *it != comment && *it != delim; ++it) {
				if (std::isspace(*it)) ws += *it;
				else {
					if (!ws.empty()) {
						i.key += ws;
						ws.clear();
					}
					i.key += *it;
				}
			}
			//advance past =
			if (it != line.end() && *it == delim) ++it;
			//ignore whitespace after = and before value
			for (; it != line.end() && std::isspace(*it); ++it);
			//the value is the rest of the line up until any comment, ignore trailing whitespace
			ws.clear();
			for (; it != line.end() && *it != comment; ++it) {
				if (std::isspace(*it)) ws += *it;
				else {
					if (!ws.empty()) {
						i.value += ws;
						ws.clear();
					}
					i.value += *it;
				}
			}
		}
		//if the line was just a comment, the item will be empty
		if (!i.empty()) items.push_back(std::move(i));
	}
	return items;
}