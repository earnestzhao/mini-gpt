#include "shared.h"
#include "common/util.h"
#include "tokenizer/constants.h"

#include <cmath>
#include <cassert>

#include <map>
#include <sstream>
#include <functional>

namespace transformer
{

////////////////////

Input::Input(embedding::Embeddings &embeddings)
    : embeddings_(embeddings),
      all_token_embeddings_(embeddings.mutable_all_token_embeddings()),
      all_position_embeddings_(embeddings.mutable_all_position_embeddings())
{

}

std::string Input::serialize_training_items() const
{
    return embeddings_.serialize()     + SERIALIZING_SPLITTER
         + gamma_X_.serialize()        + SERIALIZING_SPLITTER
         + beta_X_.serialize()         + SERIALIZING_SPLITTER
         + weight_Q_.serialize()       + SERIALIZING_SPLITTER
         + weight_K_.serialize()       + SERIALIZING_SPLITTER
         + weight_V_.serialize()       + SERIALIZING_SPLITTER
         + gamma_A_.serialize()        + SERIALIZING_SPLITTER
         + beta_A_.serialize()         + SERIALIZING_SPLITTER
         + up_projector_.serialize()   + SERIALIZING_SPLITTER
         + down_projector_.serialize() + SERIALIZING_SPLITTER
         + up_bias_.serialize()        + SERIALIZING_SPLITTER
         + down_bias_.serialize();
}

Input::DESERIALIZE_ERROR Input::deserialize_training_items(const std::string &serialized)
{
    size_t searching_start_position = 0;
    size_t searching_end_position = 0;
    std::string searching_section;

#define DESERIALIZE(object, ENAME) \
    searching_end_position = serialized.find(SERIALIZING_SPLITTER, searching_start_position); \
    if (searching_end_position == std::string::npos) \
    { \
        searching_section = serialized.substr(searching_start_position); \
    } \
    else \
    { \
        searching_section = serialized.substr(searching_start_position, searching_end_position - searching_start_position); \
    } \
    \
    if (object.deserialize(searching_section) != 0) \
    { \
        return TIDE_##ENAME##_FAILED; \
    } \
    searching_start_position = searching_end_position + SERIALIZING_SPLITTER.length();

    DESERIALIZE(embeddings_,     EMBEDDINGS)
    DESERIALIZE(gamma_X_,        GAMMA_X)
    DESERIALIZE(beta_X_,         BETA_X)
    DESERIALIZE(weight_Q_,       WEIGHT_Q)
    DESERIALIZE(weight_K_,       WEIGHT_K)
    DESERIALIZE(weight_V_,       WEIGHT_V)
    DESERIALIZE(gamma_A_,        GAMMA_A)
    DESERIALIZE(beta_A_,         BETA_A)
    DESERIALIZE(up_projector_,   UP_PROJECTOR)
    DESERIALIZE(down_projector_, DOWN_PROJECTOR)
    DESERIALIZE(up_bias_,        UP_BIAS)
    DESERIALIZE(down_bias_,      DOWN_BIAS)
#undef DESERIALIZE

    return TIDE_OK;
}

////////////////////

float Result::loss() const
{
    float loss = 0.0F;
    
    for (size_t i = 0; i < loss_.size(); i++)
    {
        loss += loss_[i].value_;
    }
    
    return loss;
}

size_t Result::generated_token_id() const
{
    auto output_token_ids_size = output_token_ids_.size();
    if (output_token_ids_size == 0)
    {
        return tokenizer::STID_INVALID;
    }
    
    return output_token_ids_[output_token_ids_size - 1];
}

std::vector<Result::Logit_Item> Result::logits(size_t index) const
{
    assert(index < softmax_.row_count());
    
    std::multimap<float, size_t, std::greater<float>> sorter;
    for (size_t j = 0; j < softmax_.col_count(); j++)
    {
        sorter.emplace(softmax_.get_element(index, j), j);
    }

    std::vector<Result::Logit_Item> logits;
    for (const auto &pair : sorter)
    {
        Logit_Item logit_item;
        logit_item.softmax_probability_ = pair.first;
        logit_item.candidate_token_id_  = pair.second;
        logits.push_back(logit_item);
    }
    
    return logits;
}

std::vector<Result::Logit_Item> Result::generated_logits() const
{
    assert(output_token_ids_.size() == softmax_.row_count());
    return logits(softmax_.row_count() - 1);
}

std::string Result::debugging_generated_logits() const
{
    const size_t DEBUGGING_LOGITS_NUMBER = 10;
    size_t logits_number_in_result = 0;

    auto logits = generated_logits();
    std::ostringstream debugging_logits_info;
    for (const auto &logit_item : logits)
    {
        debugging_logits_info << "(";
        debugging_logits_info << logit_item.candidate_token_id_;
        debugging_logits_info << ", ";
        debugging_logits_info << logit_item.softmax_probability_;
        debugging_logits_info << "), ";

        if (++ logits_number_in_result >= DEBUGGING_LOGITS_NUMBER)
        {
            break;
        }
    }

    return debugging_logits_info.str();
}

////////////////////

common::Matrix &casual_mask(common::Matrix &input_output, bool is_forward)
{
    common::Matrix result;
    casual_mask_to(input_output, result, is_forward);
    input_output = std::move(result);
    return input_output;
}

common::Matrix casual_mask_copy(const common::Matrix &input, bool is_forward)
{
    common::Matrix result;
    casual_mask_to(input, result, is_forward);
    return result;
}

void casual_mask_to(const common::Matrix &input, common::Matrix &result, bool is_forward)
{
    assert(input.valid());
    assert(input.row_count() == input.col_count());
    common::Matrix output = input;
    
    // In util's calculation logic, ZERO_EPS is kind of like 0, so, this can be treated as -INF.
    const float ATTENTION_MASK = is_forward ? - 1.0F / util::ZERO_EPS : 0.0F;
    
    for (size_t i = 0; i < output.row_count(); i++)
    {
        for (size_t j = 0; j < output.col_count(); j++)
        {
            if (j > i)
            {
                output.set_element(i, j, ATTENTION_MASK);
            }
        }
    }

    result = std::move(output);
}

////////////////////

common::Matrix &softmax(common::Matrix &input_output, bool vectors_by_row)
{
    common::Matrix result;
    softmax_to(input_output, result, vectors_by_row);
    input_output = std::move(result);
    return input_output;
}

common::Matrix softmax_copy(const common::Matrix &input, bool vectors_by_row)
{
    common::Matrix result;
    softmax_to(input, result, vectors_by_row);
    return result;
}

void softmax_to(const common::Matrix &input, common::Matrix &result, bool vectors_by_row)
{
    assert(input.valid());
    
    common::Matrix output = input;
    if (!vectors_by_row)
    {
        output.transpose();
    }
    
    for (size_t i = 0; i < output.row_count(); i++)
    {
        float sum_exp_elements = 0.0F;
        for (size_t j = 0; j < output.col_count(); j++)
        {
            float exp_element = std::exp(output.get_element(i, j));
            output.set_element(i, j, exp_element);
            sum_exp_elements += exp_element;
        }
        
        assert(!util::equal(sum_exp_elements, 0.0F));
        for (size_t j = 0; j < output.col_count(); j++)
        {
            output.set_element(i, j, output.get_element(i, j) / sum_exp_elements);
        }
    }

    if (!vectors_by_row)
    {
        output.transpose();
    }

    result = std::move(output);
}

////////////////////

common::Matrix &standardize(common::Matrix &input_output, float eps,
        Standardize_Intermediate &intermediate, bool vectors_by_row)
{
    common::Matrix result;
    standardize_to(input_output, eps, result, intermediate, vectors_by_row);
    input_output = std::move(result);
    return input_output;
}

common::Matrix standardize_copy(const common::Matrix &input, float eps,
        Standardize_Intermediate &intermediate,
        bool vectors_by_row)
{
    common::Matrix result;
    standardize_to(input, eps, result, intermediate, vectors_by_row);
    return result;
}

void standardize_to(const common::Matrix &input, float eps,
        common::Matrix &result, Standardize_Intermediate &intermediate,
        bool vectors_by_row)
{
    assert(input.valid());
    assert(!util::equal(eps, 0.0F));
    
    common::Matrix output = input;
    if (!vectors_by_row)
    {
        output.transpose();
    }
    
    intermediate.average_.resize(output.row_count());
    intermediate.variance_.resize(output.row_count());
    intermediate.denominator_.resize(output.row_count());
    
    for (size_t i = 0; i < output.row_count(); i++)
    {
        float sum_elements = 0.0F;
        for (size_t j = 0; j < output.col_count(); j++)
        {
            sum_elements += output.get_element(i, j);
        }
        
        float avg_elements = sum_elements / output.col_count();
        intermediate.average_.set_element(i, avg_elements);

        float sum_square_diff = 0.0F;
        for (size_t j = 0; j < output.col_count(); j++)
        {
            float diff = output.get_element(i, j) - avg_elements;
            sum_square_diff += diff * diff;
        }
        
        float var_elements = sum_square_diff / output.col_count();
        intermediate.variance_.set_element(i, var_elements);
        intermediate.denominator_.set_element(i, std::sqrt(var_elements + eps));

        for (size_t j = 0; j < output.col_count(); j++)
        {
            float diff = output.get_element(i, j) - avg_elements;
            float new_element = diff / intermediate.denominator_.get_element(i);
            output.set_element(i, j, new_element);
        }
    }

    if (!vectors_by_row)
    {
        output.transpose();
    }

    result = std::move(output);
}

////////////////////

float gelu_scalar(float x)
{
    return 0.5F * x * (1.0F + std::tanh(gelu_tanh_input_scalar(x)));
}

float gelu_tanh_input_scalar(float x)
{
    return std::sqrt(2.0F / M_PI) * (x + 0.044715F * x * x * x);
}

common::Matrix &gelu(common::Matrix &input_output)
{
    common::Matrix result;
    gelu_to(input_output, result);
    input_output = std::move(result);
    return input_output;
}

common::Matrix gelu_copy(const common::Matrix &input)
{
    common::Matrix result;
    gelu_to(input, result);
    return result;
}

void gelu_to(const common::Matrix &input, common::Matrix &result)
{
    assert(input.valid());
    
    common::Matrix output = input;
    for (size_t i = 0; i < output.row_count(); i++)
    {
        for (size_t j = 0; j < output.col_count(); j++)
        {
            float element_value = output.get_element(i, j);
            float element_gelu_value = gelu_scalar(element_value);
            output.set_element(i, j, element_gelu_value);
        }
    }

    result = std::move(output);
}

////////////////////

common::Matrix duplicate_vector_to_matrix(const common::Vector &v, size_t dup_count, bool vectors_by_row)
{
    common::Matrix result(dup_count, v.dim_count());
    
    for (size_t i = 0; i < dup_count; i++)
    {
        if (vectors_by_row)
        {
            v.to_vector_view().copy_to(result.row_vector(i));
        }
        else
        {
            v.to_vector_view().copy_to(result.col_vector(i));
        }
    }
    
    return result;
}

}
