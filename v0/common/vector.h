#pragma once

#include "matrix.h"

namespace common
{

class Vector
{
public:
    Vector();
    Vector(std::size_t dim_count, bool is_row_vector = true);
    Vector(const Vector &src);
    Vector(Vector &&src);

public:
    Vector &operator=(const Vector &src);
    Vector &operator=(Vector &&src);

public:
    bool valid() const;
    bool is_row_vector() const;
    bool is_col_vector() const;

public:
    size_t dim_count() const;
    void resize(std::size_t dim_count, bool to_row_vector = true);
    void transpose();

public:
    void scale(float factor);
    void plus(const Vector &other);

public:
    float get_element(size_t index) const;
    void set_element(size_t index, float element_value);
    void random_initialize(float random_range_start, float random_range_end);

public:
    std::string serialize() const;
    Matrix::MATRIX_DESERIALIZE_ERROR deserialize(const std::string &input);

public:
    operator const Matrix &();
    operator const Matrix *();
    void from_matrix(const Matrix &src);
    void from_matrix(Matrix &&src);
    void from_vector_view(const Vector_View &src, bool to_row_vector = true);
    Vector_View to_vector_view() const;

private:
    Matrix storage_;
};

}
