#include "forward.h"

#include "common/util.h"
#include "tokenizer/constants.h"

#include <cmath>
#include <cassert>

namespace transformer
{

////////////////////

common::Matrix &layernorm(common::Matrix &input_output,
        const common::Vector &gamma, const common::Vector &beta, float eps,
        bool vectors_by_row)
{
    common::Matrix result;
    layernorm_to(input_output, gamma, beta, eps, result, vectors_by_row);
    input_output = std::move(result);
    return input_output;
}

common::Matrix layernorm_copy(const common::Matrix &input,
        const common::Vector &gamma, const common::Vector &beta, float eps,
        bool vectors_by_row)
{
    common::Matrix result;
    layernorm_to(input, gamma, beta, eps, result, vectors_by_row);
    return result;
}

void layernorm_to(const common::Matrix &input,
        const common::Vector &gamma, const common::Vector &beta, float eps,
        common::Matrix &result, bool vectors_by_row)
{
    assert(input.valid());
    assert(gamma.valid() && ((vectors_by_row && gamma.is_row_vector()) || (!vectors_by_row && gamma.is_col_vector())));
    assert(beta.valid() && ((vectors_by_row && beta.is_row_vector()) || (!vectors_by_row && beta.is_col_vector())));
    assert(!util::equal(eps, 0.0F));

    common::Matrix output = input;
    if (!vectors_by_row)
    {
        output.transpose();
    }
    
    assert(gamma.dim_count() == output.col_count());
    assert(beta.dim_count() == output.col_count());
    
    Standardize_Intermediate dummy;
    standardize(output, eps, dummy);
    output.hadamard(duplicate_vector_to_matrix(gamma, output.row_count()));
    output.plus(duplicate_vector_to_matrix(beta, output.row_count()));

    if (!vectors_by_row)
    {
        output.transpose();
    }

    result = std::move(output);
}

////////////////////

common::Matrix &ffn(common::Matrix &input_output,
        const common::Matrix &up_projector, const common::Matrix &down_projector,
        const common::Vector &up_bias, const common::Vector &down_bias, bool vectors_by_row)
{
    common::Matrix result;
    ffn_to(input_output, up_projector, down_projector, up_bias, down_bias, result, vectors_by_row);
    input_output = std::move(result);
    return input_output;
}

common::Matrix ffn_copy(const common::Matrix &input,
        const common::Matrix &up_projector, const common::Matrix &down_projector,
        const common::Vector &up_bias, const common::Vector &down_bias, bool vectors_by_row)
{
    common::Matrix result;
    ffn_to(input, up_projector, down_projector, up_bias, down_bias, result, vectors_by_row);
    return result;
}

void ffn_to(const common::Matrix &input,
        const common::Matrix &up_projector, const common::Matrix &down_projector,
        const common::Vector &up_bias, const common::Vector &down_bias,
        common::Matrix &result, bool vectors_by_row)
{
    assert(input.valid());
    assert(up_projector.valid());
    assert(down_projector.valid());

    assert(up_bias.valid());
    assert(down_bias.valid());

    assert(up_projector.row_count() == down_projector.col_count());
    assert(up_projector.col_count() == down_projector.row_count());
    
    assert((vectors_by_row && up_bias.is_row_vector()) || (!vectors_by_row && up_bias.is_col_vector()));
    assert((vectors_by_row && down_bias.is_row_vector()) || (!vectors_by_row && down_bias.is_col_vector()));

    common::Matrix copied_input = input;
    common::Matrix copied_up_projector = up_projector;
    common::Matrix copied_down_projector = down_projector;

    if (!vectors_by_row)
    {
        copied_input.transpose();
        copied_up_projector.transpose();
        copied_down_projector.transpose();
    }
    
    assert(copied_up_projector.col_count() > copied_input.col_count());
    assert(copied_down_projector.col_count() == copied_input.col_count());

    auto md_up_bias   = duplicate_vector_to_matrix(up_bias,   copied_input.row_count());
    auto md_down_bias = duplicate_vector_to_matrix(down_bias, copied_input.row_count());
    common::Matrix output = gelu_copy(copied_input * copied_up_projector + md_up_bias) * copied_down_projector + md_down_bias;

    if (!vectors_by_row)
    {
        output.transpose();
    }

    result = std::move(output);
}

////////////////////

void forward(const Input &input, Result &result, Intermediate &intermediate, Forward_Mode mode, bool sampling_generation)
{
    assert(input.token_ids_.size() >= 2);
    assert(input.token_ids_[0] == tokenizer::STID_USR_TEXT_START);
    assert(mode == FM_GENERATING
        || input.token_ids_[input.token_ids_.size() - 1] == tokenizer::STID_USR_TEXT_END
        || input.token_ids_[input.token_ids_.size() - 1] == tokenizer::STID_GEN_TEXT_END);
    
    intermediate.input_token_embeddings_ = input.embeddings_.token_embeddings(input.token_ids_);
    intermediate.input_position_embeddings_ = input.embeddings_.position_embeddings(input.position_indices_);
    intermediate.input_X_ = intermediate.input_token_embeddings_ + intermediate.input_position_embeddings_;
    intermediate.layernorm_X_ = layernorm_copy(intermediate.input_X_, input.gamma_X_, input.beta_X_, input.eps_X_);

    intermediate.attention_Q_ = intermediate.layernorm_X_ * input.weight_Q_;
    intermediate.attention_K_ = intermediate.layernorm_X_ * input.weight_K_;
    intermediate.attention_V_ = intermediate.layernorm_X_ * input.weight_V_;

    // attention_K.transpose() actually changes attention_K, so here uses attention_K.row_count() as sqrt(d_K).
    intermediate.attention_A_ = intermediate.attention_Q_ * intermediate.attention_K_.transpose();
    intermediate.attention_K_.transpose();
    intermediate.attention_A_.scale(std::sqrt(1.0F / intermediate.attention_K_.col_count()));
    casual_mask(intermediate.attention_A_);
    intermediate.attention_SA_ = softmax_copy(intermediate.attention_A_);
    
    intermediate.attention_AV_ = intermediate.attention_SA_ * intermediate.attention_V_;
    intermediate.attention_AVR_ = intermediate.attention_AV_ + intermediate.input_X_;

    intermediate.layernorm_AVR_ = layernorm_copy(intermediate.attention_AVR_, input.gamma_A_, input.beta_A_, input.eps_A_);
    intermediate.ffn_ = ffn_copy(intermediate.layernorm_AVR_, input.up_projector_, input.down_projector_, input.up_bias_, input.down_bias_);
    intermediate.ffn_R_ = intermediate.ffn_ + intermediate.attention_AVR_;
    result.before_joining_vocab_ = intermediate.ffn_R_;
    
    // V0: currently, calling transpose changes the object, so, it needs to be tranposed back after calculation.
    auto &all_token_embeddings = input.embeddings_.mutable_all_token_embeddings();
    result.after_joining_vocab_ = intermediate.ffn_R_ * all_token_embeddings.transpose();
    softmax_to(result.after_joining_vocab_, result.softmax_);
    all_token_embeddings.transpose();

    result.loss_.clear();
    result.output_token_ids_.clear();

    if (mode & FM_TRAINING)
    {
        // TODO: I am not sure whether keeping it as a dialogue form is good or not.
        for (size_t i = 0; i < result.softmax_.row_count(); i++)
        {
            Result::Loss_Item loss_item;
            
            if (i < result.softmax_.row_count() - 1)
            {
                loss_item.value_ = - std::log(result.softmax_.get_element(i, input.token_ids_[i + 1]));
                loss_item.on_token_id_ = input.token_ids_[i + 1];
            }
            else
            {
                if (input.token_ids_[i] == tokenizer::STID_USR_TEXT_END)
                {
                    loss_item.value_ = - std::log(result.softmax_.get_element(i, tokenizer::STID_GEN_TEXT_START));
                    loss_item.on_token_id_ = tokenizer::STID_GEN_TEXT_START;
                }
                else
                {
                    loss_item.value_ = - std::log(result.softmax_.get_element(i, tokenizer::STID_GEN_TEXT_END));
                    loss_item.on_token_id_ = tokenizer::STID_GEN_TEXT_END;
                }
            }
            
            result.loss_.push_back(loss_item);
        }
    }

    if (mode & FM_GENERATING)
    {
        for (size_t i = 0; i < result.softmax_.row_count(); i++)
        {
            if (sampling_generation)
            {
                auto current_row_logits = result.logits(i);
                assert(current_row_logits.size() > 0);
                
                size_t selected_token_index = 0;
                auto selector = util::random(0.0F, 1.0F);
                
                float sum_softmax_probability = 0.0F;
                for (size_t j = current_row_logits.size(); j-- > 0;)
                {
                    sum_softmax_probability += current_row_logits[j].softmax_probability_;

                    if (tokenizer::unexpected_token_id(current_row_logits[j].candidate_token_id_))
                    {
                        continue;
                    }

                    if (selector < sum_softmax_probability)
                    {
                        selected_token_index = j;
                        break;
                    }
                }

                if (selected_token_index == 0 && tokenizer::unexpected_token_id(current_row_logits[0].candidate_token_id_))
                {
                    for (size_t j = 0; j < current_row_logits.size(); j++)
                    {
                        if (!tokenizer::unexpected_token_id(current_row_logits[j].candidate_token_id_))
                        {
                            selected_token_index = j;
                            break;
                        }
                    }
                }
                
                result.output_token_ids_.push_back(current_row_logits[selected_token_index].candidate_token_id_);
            }
            else
            {
                size_t max_element_index = tokenizer::EXPECTED_TOKEN_START_ID;
                float  max_element_value = result.softmax_.get_element(i, max_element_index);

                for (size_t j = max_element_index + 1; j < result.softmax_.col_count(); j++)
                {
                    float element_value = result.softmax_.get_element(i, j);
                    if (!tokenizer::unexpected_token_id(j) && util::greater_than(element_value, max_element_value))
                    {
                        max_element_index = j;
                        max_element_value = element_value;
                    }
                }
                
                result.output_token_ids_.push_back(max_element_index);
            }
        }
    }
}

}
