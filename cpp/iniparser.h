#pragma once
// This is free and unencumbered software released into the public domain.

/*
Parse ini files.
Sections and values are optional.
Leading and trailing whitespace are ignored, whitespace around the = is discared
Handles \r, \n, and \r\n line endings correctly.

Example format:
[Section]
item=value ; a comment
*/
#include <string>
#include <vector>
#include <fstream>

namespace ini {
	struct item {
		std::string section;
		std::string key;
		std::string value;
		bool empty() {
			return section.empty() && key.empty() && value.empty();
		}
	};
	//Extract a line, handling \r, \n, and \r\n correctly
	std::istream& getline_s(std::istream& is, std::string &line);
	std::vector<item> load(const std::string &fname, char comment=';', char delim='=');
	std::vector<item> load(std::istream &in, char comment = ';', char delim = '=');
}