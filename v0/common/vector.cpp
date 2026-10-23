#include "vector.h"
#include <cassert>

namespace common
{

////////////////////

Vector::Vector() : storage_()
{

}

Vector::Vector(std::size_t dim_count, bool is_row_vector) : storage_(1, dim_count)
{
    if (!is_row_vector)
    {
        storage_.transpose();
    }
}

Vector::Vector(const Vector &src) : storage_(src.storage_)
{

}

Vector::Vector(Vector &&src) : storage_(std::move(src))
{

}

////////////////////

Vector &Vector::operator=(const Vector &src)
{
    storage_ = src.storage_;
    return *this;
}

Vector &Vector::operator=(Vector &&src)
{
    storage_ = std::move(src.storage_);
    return *this;
}

////////////////////

bool Vector::valid() const
{
    return storage_.valid() && (storage_.row_count() == 1 || storage_.col_count() == 1);
}

bool Vector::is_row_vector() const
{
    assert(valid());
    return storage_.row_count() == 1;
}

bool Vector::is_col_vector() const
{
    assert(valid());
    return storage_.col_count() == 1;
}

////////////////////

size_t Vector::dim_count() const
{
    if (is_col_vector())
    {
        return storage_.row_count();
    }

    return storage_.col_count();
}

void Vector::resize(std::size_t dim_count, bool to_row_vector)
{
    if (!to_row_vector)
    {
        storage_.resize(dim_count, 1);
    }
    else
    {
        storage_.resize(1, dim_count);
    }
}

void Vector::transpose()
{
    storage_.transpose();
}

////////////////////

void Vector::scale(float factor)
{
    storage_.scale(factor);
}

void Vector::plus(const Vector &other)
{
    storage_.plus(other.storage_);
}

////////////////////

float Vector::get_element(size_t index) const
{
    if (is_col_vector())
    {
        return storage_.get_element(index, 0);
    }
    
    return storage_.get_element(0, index);
}

void Vector::set_element(size_t index, float element_value)
{
    if (is_col_vector())
    {
        storage_.set_element(index, 0, element_value);
    }
    else
    {
        storage_.set_element(0, index, element_value);
    }
}

////////////////////

std::string Vector::serialize() const
{
    return storage_.serialize();
}

Matrix::MATRIX_DESERIALIZE_ERROR Vector::deserialize(const std::string &input)
{
    return storage_.deserialize(input);
}

void Vector::random_initialize(float random_range_start, float random_range_end)
{
    storage_.random_initialize(random_range_start, random_range_end);
}

////////////////////

Vector::operator const Matrix &()
{
    return storage_;
}

Vector::operator const Matrix *()
{
    return &storage_;
}

void Vector::from_matrix(const Matrix &src)
{
    assert(src.valid());
    assert(src.row_count() == 1 || src.col_count() == 1);
    storage_ = src;
}

void Vector::from_matrix(Matrix &&src)
{
    assert(src.valid());
    assert(src.row_count() == 1 || src.col_count() == 1);
    storage_ = std::move(src);
}

void Vector::from_vector_view(const Vector_View &src, bool to_row_vector)
{
    assert(src.valid());
    storage_ = Matrix(1, src.dim_count());
    if (to_row_vector)
    {
        storage_.write_row_vector_from(0, src);
    }
    else
    {
        storage_.transpose();
        storage_.write_col_vector_from(0, src);
    }
}

Vector_View Vector::to_vector_view() const
{
    if (is_col_vector())
    {
        return storage_.col_vector(0);
    }

    return storage_.row_vector(0);
}

}
