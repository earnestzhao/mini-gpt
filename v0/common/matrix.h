#pragma once

#include <cstddef>
#include <string>
#include "vector_view.h"

namespace common
{

class Matrix
{
public:
    Matrix();
    Matrix(std::size_t row_count, std::size_t col_count);
    Matrix(const Matrix &src);
    Matrix(Matrix &&src);
    virtual ~Matrix();

public:
    void unique_initialize(float element_value);
    void random_initialize(float random_range_start, float random_range_end);
    void resize(std::size_t row_count, std::size_t col_count);
    void to_zero();

public:
    static bool have_same_size(const Matrix &lhs, const Matrix &rhs);
    static bool are_able_to_matmul(const Matrix &lhs, const Matrix &rhs);

    static void compute_plus(const Matrix &lhs, const Matrix &rhs, Matrix &result);
    static void compute_scale(const Matrix &lhs, float rhs, Matrix &result);
    static void compute_matmul(const Matrix &lhs, const Matrix &rhs, Matrix &result);
    static void compute_hadamard(const Matrix &lhs, const Matrix &rhs, Matrix &result);
    static void compute_transpose(const Matrix &src, Matrix &transposed);

    static void compute_row_concat(const Matrix &lhs, const Matrix &rhs, Matrix &result);
    static void compute_col_concat(const Matrix &lhs, const Matrix &rhs, Matrix &result);

public:
    Matrix &transpose();
    Matrix transpose_copy() const;
    void transpose_to(Matrix &result) const;
    
    Matrix &plus(const Matrix &other);
    Matrix plus_copy(const Matrix &other) const;
    void plus_to(const Matrix &other, Matrix &result) const;

    Matrix &scale(float factor);
    Matrix scale_copy(float factor) const;
    void scale_to(float factor, Matrix &result) const;
    
    Matrix &matmul(const Matrix &other);
    Matrix matmul_copy(const Matrix &other) const;
    void matmul_to(const Matrix &other, Matrix &result) const;

    Matrix &hadamard(const Matrix &other);
    Matrix hadamard_copy(const Matrix &other) const;
    void hadamard_to(const Matrix &other, Matrix &result) const;

public:
    Matrix operator+(const Matrix &rhs) const;
    Matrix operator*(const Matrix &rhs) const;
    Matrix &operator=(const Matrix &rhs);
    Matrix &operator=(Matrix &&rhs);

public:
    bool valid() const;
    std::size_t row_count() const;
    std::size_t col_count() const;
    float get_element(std::size_t row_index, std::size_t col_index) const;
    void set_element(std::size_t row_index, std::size_t col_index, float element_value);
    void plus_element(std::size_t row_index, std::size_t col_index, float element_value);

public:
    Vector_View row_vector(std::size_t row_index) const;
    Vector_View col_vector(std::size_t col_index) const;
    void write_row_vector_from(std::size_t row_index, const Vector_View &vector);
    void write_col_vector_from(std::size_t col_index, const Vector_View &vector);

public:
    std::string serialize() const;
    enum MATRIX_DESERIALIZE_ERROR
    {
        MDE_OK = 0,
        MDE_EMPTY_INPUT = 1,
        MDE_INVALID_INPUT_FORMAT  = -1,
        MDE_INVALID_COLUMN_COUNT  = -2,
        MDE_INVALID_ELEMENT_VALUE = -3,
    };
    MATRIX_DESERIALIZE_ERROR deserialize(const std::string &input);

private:
    void create(std::size_t row_count, std::size_t col_count);
    void initialize_to_invalid();
    void destroy();

private:
    void move_from(Matrix &&src);
    void copy_from(const Matrix &src);

private:
    std::size_t row_count_ = 0;
    std::size_t col_count_ = 0;
    bool transposed_ = false;
    float *storage_  = nullptr;
};

}
