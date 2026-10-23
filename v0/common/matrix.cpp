#include "matrix.h"
#include "util.h"

#include <utility>
#include <cassert>
#include <sstream>
#include <algorithm>

namespace common
{

////////////////////

Matrix::Matrix()
{
    initialize_to_invalid();
}

Matrix::Matrix(size_t row_count, size_t col_count)
{
    create(row_count, col_count);
}

Matrix::Matrix(const Matrix &src)
{
    copy_from(src);
}

Matrix::Matrix(Matrix &&src)
{
    move_from(std::move(src));
}

Matrix::~Matrix()
{
    destroy();
}

////////////////////

void Matrix::unique_initialize(float element_value)
{
    assert(valid());
    std::fill_n(storage_, row_count_ * col_count_, element_value);
}

void Matrix::random_initialize(float random_range_start, float random_range_end)
{
    assert(valid());

    for (size_t i = 0; i < row_count_ * col_count_; i++)
    {
        storage_[i] = util::random(random_range_start, random_range_end);
    }
}

void Matrix::resize(size_t row_count, size_t col_count)
{
    if (!valid() || this->row_count() != row_count || this->col_count() != col_count)
    {
        destroy();
        create(row_count, col_count);
    }
}

void Matrix::to_zero()
{
    unique_initialize(0.0F);
}

////////////////////

bool Matrix::have_same_size(const Matrix &lhs, const Matrix &rhs)
{
    assert(lhs.valid() && rhs.valid());
    return lhs.row_count() == rhs.row_count() && lhs.col_count() == rhs.col_count();
}

bool Matrix::are_able_to_matmul(const Matrix &lhs, const Matrix &rhs)
{
    assert(lhs.valid() && rhs.valid());
    return lhs.col_count() == rhs.row_count();
}

void Matrix::compute_plus(const Matrix &lhs, const Matrix &rhs, Matrix &result)
{
    assert(lhs.valid() && rhs.valid());
    assert(have_same_size(lhs, rhs));
    assert(&lhs != &result);
    assert(&rhs != &result);

    auto result_row_count = lhs.row_count();
    auto result_col_count = lhs.col_count();
    result.resize(result_row_count, result_col_count);

    for (size_t i = 0; i < result_row_count; i++)
    {
        for (size_t j = 0; j < result_col_count; j++)
        {
            result.set_element(i, j, lhs.get_element(i, j) + rhs.get_element(i, j));
        }
    }
}

void Matrix::compute_scale(const Matrix &lhs, float rhs, Matrix &result)
{
    assert(lhs.valid());

    auto result_row_count = lhs.row_count();
    auto result_col_count = lhs.col_count();
    result.resize(result_row_count, result_col_count);

    for (size_t i = 0; i < result_row_count; i++)
    {
        for (size_t j = 0; j < result_col_count; j++)
        {
            result.set_element(i, j, lhs.get_element(i, j) * rhs);
        }
    }
}

void Matrix::compute_matmul(const Matrix &lhs, const Matrix &rhs, Matrix &result)
{
    assert(lhs.valid() && rhs.valid());
    assert(are_able_to_matmul(lhs, rhs));
    assert(&lhs != &result);
    assert(&rhs != &result);
    
    auto result_row_count = lhs.row_count();
    auto result_col_count = rhs.col_count();
    auto traversal_element_count = lhs.col_count();
    
    result.resize(result_row_count, result_col_count);
    result.to_zero();

    for (size_t i = 0; i < result_row_count; i++)
    {
        for (size_t j = 0; j < result_col_count; j++)
        {
            for (size_t k = 0; k < traversal_element_count; k++)
            {
                result.plus_element(i, j, lhs.get_element(i, k) * rhs.get_element(k, j));
            }
        }
    }
}

void Matrix::compute_hadamard(const Matrix &lhs, const Matrix &rhs, Matrix &result)
{
    assert(lhs.valid() && rhs.valid());
    assert(have_same_size(lhs, rhs));
    assert(&lhs != &result);
    assert(&rhs != &result);

    auto result_row_count = lhs.row_count();
    auto result_col_count = lhs.col_count();
    result.resize(result_row_count, result_col_count);

    for (size_t i = 0; i < result_row_count; i++)
    {
        for (size_t j = 0; j < result_col_count; j++)
        {
            result.set_element(i, j, lhs.get_element(i, j) * rhs.get_element(i, j));
        }
    }
}

void Matrix::compute_transpose(const Matrix &src, Matrix &transposed)
{
    assert(src.valid());
    if (&src == &transposed)
    {
        transposed.transpose();
        return;
    }
    
    auto result_row_count = src.col_count();
    auto result_col_count = src.row_count();
    transposed.resize(result_row_count, result_col_count);
    
    for (size_t i = 0; i < result_row_count; i++)
    {
        for (size_t j = 0; j < result_col_count; j++)
        {
            transposed.set_element(i, j, src.get_element(j, i));
        }
    }
}

void Matrix::compute_row_concat(const Matrix &lhs, const Matrix &rhs, Matrix &result)
{
    assert(lhs.valid() && rhs.valid());
    assert(lhs.col_count() == rhs.col_count());
    
    auto result_row_count = lhs.row_count() + rhs.row_count();
    auto result_col_count = lhs.col_count();
    result.resize(result_row_count, result_col_count);
    
    for (size_t i = 0; i < result_row_count; i++)
    {
        const Matrix *current_operand    = i < lhs.row_count() ? &lhs : &rhs;
        size_t current_operand_row_index = i < lhs.row_count() ? i : i - lhs.row_count();
        
        for (size_t j = 0; j < result_col_count; j++)
        {
            result.set_element(i, j, current_operand->get_element(current_operand_row_index, j));
        }
    }
}

void Matrix::compute_col_concat(const Matrix &lhs, const Matrix &rhs, Matrix &result)
{
    assert(lhs.valid() && rhs.valid());
    assert(lhs.row_count() == rhs.row_count());

    auto result_row_count = lhs.row_count();
    auto result_col_count = lhs.col_count() + rhs.col_count();
    result.resize(result_row_count, result_col_count);

    for (size_t i = 0; i < result_row_count; i++)
    {
        for (size_t j = 0; j < result_col_count; j++)
        {
            const Matrix *current_operand    = j < lhs.col_count() ? &lhs : &rhs;
            size_t current_operand_col_index = j < lhs.col_count() ? j : j - lhs.col_count();
            result.set_element(i, j, current_operand->get_element(i, current_operand_col_index));
        }
    }
}

////////////////////

Matrix &Matrix::transpose()
{
    assert(valid());
    transposed_ = !transposed_;
    return *this;
}

Matrix Matrix::transpose_copy() const
{
    Matrix result;
    compute_transpose(*this, result);
    return result;
}

void Matrix::transpose_to(Matrix &result) const
{
    compute_transpose(*this, result);
}

Matrix &Matrix::plus(const Matrix &other)
{
    Matrix result;
    compute_plus(*this, other, result);
    *this = std::move(result);
    return *this;
}

Matrix Matrix::plus_copy(const Matrix &other) const
{
    Matrix result;
    compute_plus(*this, other, result);
    return result;
}

void Matrix::plus_to(const Matrix &other, Matrix &result) const
{
    compute_plus(*this, other, result);
}

Matrix &Matrix::scale(float factor)
{
    Matrix result;
    compute_scale(*this, factor, result);
    *this = std::move(result);
    return *this;
}

Matrix Matrix::scale_copy(float factor) const
{
    Matrix result;
    compute_scale(*this, factor, result);
    return result;
}

void Matrix::scale_to(float factor, Matrix &result) const
{
    compute_scale(*this, factor, result);
}

Matrix &Matrix::matmul(const Matrix &other)
{
    Matrix result;
    compute_matmul(*this, other, result);
    *this = std::move(result);
    return *this;
}

Matrix Matrix::matmul_copy(const Matrix &other) const
{
    Matrix result;
    compute_matmul(*this, other, result);
    return result;
}

void Matrix::matmul_to(const Matrix &other, Matrix &result) const
{
    compute_matmul(*this, other, result);
}

Matrix &Matrix::hadamard(const Matrix &other)
{
    Matrix result;
    compute_hadamard(*this, other, result);
    *this = std::move(result);
    return *this;
}

Matrix Matrix::hadamard_copy(const Matrix &other) const
{
    Matrix result;
    compute_hadamard(*this, other, result);
    return result;
}

void Matrix::hadamard_to(const Matrix &other, Matrix &result) const
{
    compute_hadamard(*this, other, result);
}

////////////////////
    
Matrix Matrix::operator+(const Matrix &rhs) const
{
    return plus_copy(rhs);
}

Matrix Matrix::operator*(const Matrix &rhs) const
{
    return matmul_copy(rhs);
}

Matrix &Matrix::operator=(const Matrix &rhs)
{
    copy_from(rhs);
    return *this;
}

Matrix &Matrix::operator=(Matrix &&rhs)
{
    move_from(std::move(rhs));
    return *this;
}

////////////////////

bool Matrix::valid() const
{
    return row_count_ > 0 && col_count_ > 0 && storage_ != nullptr;
}

size_t Matrix::row_count() const
{
    assert(valid());
    return transposed_ ? col_count_ : row_count_;
}

size_t Matrix::col_count() const
{
    assert(valid());
    return transposed_ ? row_count_ : col_count_;
}

float Matrix::get_element(size_t row_index, size_t col_index) const
{
    assert(valid());
    assert(row_index < row_count());
    assert(col_index < col_count());
    return transposed_ ? storage_[col_index * col_count_ + row_index] : storage_[row_index * col_count_ + col_index];
}

void Matrix::set_element(size_t row_index, size_t col_index, float element_value)
{
    assert(valid());
    assert(row_index < row_count());
    assert(col_index < col_count());
    
    if (transposed_)
    {
        storage_[col_index * col_count_ + row_index] = element_value;
    }
    else
    {
        storage_[row_index * col_count_ + col_index] = element_value;
    }
}

void Matrix::plus_element(size_t row_index, size_t col_index, float element_value)
{
    set_element(row_index, col_index, get_element(row_index, col_index) + element_value);
}

////////////////////

Vector_View Matrix::row_vector(size_t row_index) const
{
    assert(valid());
    assert(row_index < row_count());

    float *data   = nullptr;
    size_t length = col_count();
    size_t stride = 0;

    if (transposed_)
    {
        data   = &storage_[row_index];
        stride = col_count_;
    }
    else
    {
        data   = &storage_[row_index * col_count_];
        stride = 1;
    }
    
    return Vector_View(data, length, stride);
}

Vector_View Matrix::col_vector(size_t col_index) const
{
    assert(valid());
    assert(col_index < col_count());

    float *data   = nullptr;
    size_t length = row_count();
    size_t stride = 0;

    if (transposed_)
    {
        data   = &storage_[col_index * col_count_];
        stride = 1;
    }
    else
    {
        data   = &storage_[col_index];
        stride = col_count_;
    }
    
    return Vector_View(data, length, stride);
}

void Matrix::write_row_vector_from(size_t row_index, const Vector_View &vector)
{
    assert(valid());
    assert(row_index < row_count());
    vector.copy_to(row_vector(row_index));
}

void Matrix::write_col_vector_from(size_t col_index, const Vector_View &vector)
{
    assert(valid());
    assert(col_index < col_count());
    vector.copy_to(col_vector(col_index));
}

////////////////////

std::string Matrix::serialize() const
{
    assert(valid());
    
    std::ostringstream result_stream;
    auto result_row_count = row_count();
    auto result_col_count = col_count();

    for (size_t i = 0; i < result_row_count; i++)
    {
        for (size_t j = 0; j < result_col_count; j++)
        {
            result_stream << get_element(i, j) << ", ";
        }
        
        result_stream << "\n";
    }
    
    return result_stream.str();
}

Matrix::MATRIX_DESERIALIZE_ERROR Matrix::deserialize(const std::string &input)
{
    if (input.length() == 0)
    {
        return MDE_EMPTY_INPUT;
    }
    
    Matrix deserialized_result;
    size_t row_count = 0;
    size_t col_count = 0;

    for (size_t i = 0, col_count_in_row = 0; i < input.length(); i++)
    {
        if (input[i] == '\n')
        {
            if (row_count == 0)
            {
                col_count = col_count_in_row;
            }
            else
            {
                if (col_count_in_row != col_count)
                {
                    deserialized_result.destroy();
                    return MDE_INVALID_COLUMN_COUNT;
                }
            }
            
            row_count ++;
            col_count_in_row = 0;
        }
        else if (input[i] == ',')
        {
            col_count_in_row ++;
        }
    }
    
    deserialized_result.resize(row_count, col_count);
    std::string element_str_value;
    size_t row_index = 0;
    size_t col_index = 0;
    size_t converted_count = 0;
    
    for (size_t i = 0; i < input.length(); i++)
    {
        if (input[i] == '\n')
        {
            row_index ++;
            col_index = 0;
        }
        else if (input[i] == ',')
        {
            deserialized_result.set_element(row_index, col_index, (float)std::stod(element_str_value, &converted_count));
            if (converted_count != element_str_value.length())
            {
                deserialized_result.destroy();
                return MDE_INVALID_ELEMENT_VALUE;
            }
            
            element_str_value = "";
            col_index ++;
        }
        else if (input[i] == ' ')
        {

        }
        else if ((input[i] >= '0' && input[i] <= '9') || input[i] == '-' || input[i] == '.' || input[i] == 'e' || input[i] == 'E')
        {
            element_str_value.push_back(input[i]);
        }
        else
        {
            deserialized_result.destroy();
            return MDE_INVALID_INPUT_FORMAT;
        }
    }

    *this = std::move(deserialized_result);
    return MDE_OK;
}

////////////////////

void Matrix::create(size_t row_count, size_t col_count)
{
    if (row_count == 0 || col_count == 0)
    {
        initialize_to_invalid();
        return;
    }
    
    storage_ = new float[row_count * col_count];
    row_count_  = row_count;
    col_count_  = col_count;
    transposed_ = false;
}

void Matrix::initialize_to_invalid()
{
    row_count_  = 0;
    col_count_  = 0;
    transposed_ = false;
    storage_    = nullptr;
}

void Matrix::destroy()
{
    delete[] storage_;
    initialize_to_invalid();
}

////////////////////

void Matrix::move_from(Matrix &&src)
{
    destroy();
    row_count_  = src.row_count_;
    col_count_  = src.col_count_;
    transposed_ = src.transposed_;
    storage_    = src.storage_;
    src.initialize_to_invalid();
}

void Matrix::copy_from(const Matrix &src)
{
    if (this == &src)
    {
        return;
    }
    
    resize(src.row_count(), src.col_count());

    for (size_t i = 0; i < this->row_count(); i++)
    {
        for (size_t j = 0; j < this->col_count(); j++)
        {
            this->set_element(i, j, src.get_element(i, j));
        }
    }
}

////////////////////

}
