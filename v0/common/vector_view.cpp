#include "vector_view.h"
#include <cstring>
#include <cassert>

namespace common
{

Vector_View::Vector_View(float *data, size_t length, size_t stride) : data_(data), length_(length), stride_(stride)
{

}

bool Vector_View::valid() const
{
    return data_ != nullptr && length_ != 0 && stride_ != 0;
}

size_t Vector_View::dim_count() const
{
    assert(valid());
    return length_;
}

void Vector_View::copy_to(float *dst, size_t dst_length, size_t dst_stride) const
{
    assert(valid());
    assert(this->length_ == dst_length);

    if (stride_ == 1 && dst_stride == 1)
    {
        std::memcpy((char *)dst, (char *)this->data_, this->length_ * sizeof(float));
    }
    else
    {
        for (size_t src_index = 0, dst_index = 0; src_index < this->length_; src_index ++, dst_index ++)
        {
            dst[dst_index * dst_stride] = this->data_[src_index * stride_];
        }
    }
}

void Vector_View::copy_to(const Vector_View &dst) const
{
    copy_to(dst.data_, dst.length_, dst.stride_);
}

}
