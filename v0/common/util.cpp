#include "util.h"
#include <random>
#include <fstream>
#include <iterator>

namespace util
{

void swap(float &lhs, float &rhs)
{
    float helper = lhs;
    lhs = rhs;
    rhs = helper;
}

bool equal(float lhs, float rhs)
{
    if (fabs(lhs - rhs) <= ZERO_EPS)
    {
        return true;
    }
    
    return false;
}

bool less_than(float lhs, float rhs)
{
    if (equal(lhs, rhs))
    {
        return false;
    }

    return lhs < rhs;
}

bool greater_than(float lhs, float rhs)
{
    if (equal(lhs, rhs))
    {
        return false;
    }
    
    return lhs > rhs;
}

float random(float range_start, float range_end)
{
    if (greater_than(range_start, range_end))
    {
        swap(range_start, range_end);
    }
    
    static std::random_device random_device;
    static std::mt19937 generator(random_device());
    std::uniform_real_distribution<float> distributor(range_start, range_end);
    return distributor(generator);
}

int dump_string_to_file(const std::string &str, const std::string &path_filename)
{
    std::ofstream file(path_filename);

    if (!file.is_open())
    {
        return -1;
    }

    file << str;

    if (!file)
    {
        return -2;
    }

    return 0;
}

int read_string_from_file(const std::string &path_filename, std::string &str)
{
    std::ifstream file(path_filename);

    if (!file.is_open())
    {
        return -1;
    }
    
    str.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return 0;
}

}
