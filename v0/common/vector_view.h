#pragma once

#include <cstddef>

namespace common
{

class Vector_View
{
public:
    Vector_View() = delete;
    Vector_View(float *data, size_t length, size_t stride = 1);

public:
    bool valid() const;
    size_t dim_count() const;

public:
    void copy_to(float *dst, size_t dst_length, size_t dst_stride) const;
    void copy_to(const Vector_View &dst) const;

private:
    float *data_ = nullptr;
    size_t length_ = 0;
    size_t stride_ = 1;
};

}
