#pragma once

#include <string>

namespace util
{

// Constants
const float ZERO_EPS  = 1e-6;
const float SMALL_EPS = ZERO_EPS * 10;

// Swap 2 float numbers
void swap(float &lhs, float &rhs);

// Compare 2 float numbers
bool equal(float lhs, float rhs);
bool less_than(float lhs, float rhs);
bool greater_than(float lhs, float rhs);

// Generate random float number between range_start and range_end.
float random(float range_start, float range_end);

// String-file operation functions
int dump_string_to_file(const std::string &str, const std::string &path_filename);
int read_string_from_file(const std::string &path_filename, std::string &str);

}
