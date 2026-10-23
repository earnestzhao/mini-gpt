#include "backward.h"

#include <cmath>
#include <cassert>

namespace transformer
{

float d_gelu_scalar(float x)
{
    float tanh_value = std::tanh(gelu_tanh_input_scalar(x));
    return 0.5F + 0.5F * tanh_value + 0.5 * x * (1 - tanh_value * tanh_value) * d_gelu_tanh_input_scalar(x);
}

float d_gelu_tanh_input_scalar(float x)
{
    return std::sqrt(2.0F / M_PI) * (1 + 3 * 0.044715F * x * x);
}

////////////////////

void d_softmax(const common::Matrix &result, const common::Matrix &d_result, common::Matrix &d_input)
{
    assert(result.valid());
    assert(d_result.valid());
    assert(result.row_count() == d_result.row_count());
    assert(result.col_count() == d_result.col_count());

    d_input.resize(d_result.row_count(), d_result.col_count());

    for (size_t i = 0; i < result.row_count(); i++)
    {
        float multi_sum = 0.0F;
        for (size_t j = 0; j < result.col_count(); j++)
        {
            multi_sum += result.get_element(i, j) * d_result.get_element(i, j);
        }

        for (size_t j = 0; j < result.col_count(); j++)
        {
            d_input.set_element(i, j, result.get_element(i, j) * (d_result.get_element(i, j) - multi_sum));
        }
    }
}

void d_layernorm(const common::Matrix &d_result, const common::Matrix &input, const common::Vector &gamma, float eps,
        common::Matrix &d_input, common::Vector &d_gamma, common::Vector &d_beta)
{
    assert(input.valid());
    assert(gamma.valid());
    assert(d_result.valid());

    assert(input.row_count() == d_result.row_count());
    assert(input.col_count() == d_result.col_count());
    assert(input.col_count() == gamma.dim_count());

    Standardize_Intermediate standardize_intermediate;
    d_input.resize(input.row_count(), input.col_count());
    d_gamma.resize(gamma.dim_count());
    d_beta.resize(gamma.dim_count());

    common::Matrix standardized_input = standardize_copy(input, eps, standardize_intermediate);
    for (size_t i = 0; i < d_input.row_count(); i++)
    {
        for (size_t j = 0; j < d_input.col_count(); j++)
        {
            float multi_sum = 0.0F;
            for (size_t l = 0; l < d_result.col_count(); l++)
            {
                multi_sum += d_result.get_element(i, l) * gamma.get_element(l) * (standardized_input.get_element(i, l) * standardized_input.get_element(i, j) + 1);
            }

            d_input.set_element(i, j, (d_result.get_element(i, j) * d_result.col_count() * gamma.get_element(j) - multi_sum) / (d_result.col_count() * standardize_intermediate.denominator_.get_element(i)));
        }
    }

    for (size_t j = 0; j < d_gamma.dim_count(); j++)
    {
        d_gamma.set_element(j, 0.0F);
        d_beta.set_element(j, 0.0F);

        for (size_t k = 0; k < d_result.row_count(); k++)
        {
            d_gamma.set_element(j, d_gamma.get_element(j) + d_result.get_element(k, j) * standardized_input.get_element(k, j));
            d_beta.set_element(j, d_beta.get_element(j) + d_result.get_element(k, j));
        }
    }
}

////////////////////

void backward(Input &input, const Result &result, const Intermediate &intermediate,
        Input_Gradients *input_gradients, bool update, float learning_rate)
{
    assert(!update || (learning_rate > 0.0F && learning_rate < 1.0F));

    common::Matrix d_result_softmax(result.softmax_.row_count(), result.softmax_.col_count());
    d_result_softmax.unique_initialize(0.0F);
    for (size_t i = 0; i < result.loss_.size(); i++)
    {
        d_result_softmax.set_element(i, result.loss_[i].on_token_id_, - 1 / result.softmax_.get_element(i, result.loss_[i].on_token_id_));
    }
    
    common::Matrix d_result_after_joining_vocab;
    d_softmax(result.softmax_, d_result_softmax, d_result_after_joining_vocab);
    
    common::Matrix d_before_joining_vocab = d_result_after_joining_vocab * input.embeddings_.all_token_embeddings();
    common::Matrix d_all_token_embeddings = d_result_after_joining_vocab.transpose() * result.before_joining_vocab_;
    d_result_after_joining_vocab.transpose();
    
    common::Matrix d_attention_AVR = d_before_joining_vocab;
    const common::Matrix &d_ffn = d_before_joining_vocab;
    common::Matrix d_gelu = d_ffn * input.down_projector_.transpose();
    input.down_projector_.transpose();
    
    common::Matrix d_layernorm_AVR(intermediate.layernorm_AVR_.row_count(), intermediate.layernorm_AVR_.col_count());
    common::Matrix gelu_parameter = intermediate.layernorm_AVR_ * input.up_projector_ + duplicate_vector_to_matrix(input.up_bias_, intermediate.layernorm_AVR_.row_count());
    for (size_t i = 0; i < d_layernorm_AVR.row_count(); i++)
    {
        for (size_t j = 0; j < d_layernorm_AVR.col_count(); j++)
        {
            d_layernorm_AVR.set_element(i, j, 0.0F);
            
            for (size_t l = 0; l < input.up_projector_.col_count(); l++)
            {
                d_layernorm_AVR.set_element(i, j, d_layernorm_AVR.get_element(i, j) + d_gelu.get_element(i, l) * input.up_projector_.get_element(j, l) * d_gelu_scalar(gelu_parameter.get_element(i, l)));
            }
        }
    }

    common::Matrix d_up_projector(input.up_projector_.row_count(), input.up_projector_.col_count());
    for (size_t i = 0; i < d_up_projector.row_count(); i++)
    {
        for (size_t j = 0; j < d_up_projector.col_count(); j++)
        {
            d_up_projector.set_element(i, j, 0.0F);

            for (size_t k = 0; k < intermediate.layernorm_AVR_.row_count(); k++)
            {
                d_up_projector.set_element(i, j, d_up_projector.get_element(i, j) + d_gelu.get_element(k, j) * intermediate.layernorm_AVR_.get_element(k, i) * d_gelu_scalar(gelu_parameter.get_element(k, j)));
            }
        }
    }

    common::Vector d_up_bias(input.up_bias_.dim_count());
    for (size_t j = 0; j < d_up_bias.dim_count(); j++)
    {
        d_up_bias.set_element(j, 0.0F);

        for (size_t k = 0; k < d_gelu.row_count(); k++)
        {
            d_up_bias.set_element(j, d_up_bias.get_element(j) + d_gelu.get_element(k, j) * d_gelu_scalar(gelu_parameter.get_element(k, j)));
        }
    }

    common::Matrix d_down_projector = gelu(gelu_parameter).transpose() * d_ffn;
    common::Vector d_down_bias(input.down_bias_.dim_count());
    for (size_t j = 0; j < d_ffn.col_count(); j++)
    {
        d_down_bias.set_element(j, 0.0F);

        for (size_t i = 0; i < d_ffn.row_count(); i++)
        {
            d_down_bias.set_element(j, d_down_bias.get_element(j) + d_ffn.get_element(i, j));
        }
    }

    common::Matrix d_attention_AVR_from_layernorm;
    common::Vector d_gamma_A;
    common::Vector d_beta_A;
    d_layernorm(d_layernorm_AVR, intermediate.attention_AVR_, input.gamma_A_, input.eps_A_,
            d_attention_AVR_from_layernorm, d_gamma_A, d_beta_A);
    d_attention_AVR.plus(d_attention_AVR_from_layernorm);
    
    common::Matrix d_attention_SA = d_attention_AVR * intermediate.attention_V_.transpose_copy();
    common::Matrix d_attention_V = intermediate.attention_SA_.transpose_copy() * d_attention_AVR;

    common::Matrix d_attention_A;
    d_softmax(intermediate.attention_SA_, d_attention_SA, d_attention_A);
    d_attention_A.scale(1.0F / std::sqrt(intermediate.attention_K_.col_count()));
    
    common::Matrix d_attention_QK = d_attention_A;
    casual_mask(d_attention_QK, false); // Bug: adding is_forward = false to fix it.
    common::Matrix d_attention_Q = d_attention_QK * intermediate.attention_K_;
    common::Matrix d_attention_K = d_attention_QK.transpose() * intermediate.attention_Q_;

    auto transposed_layernorm_X = intermediate.layernorm_X_.transpose_copy();
    common::Matrix d_weight_Q = transposed_layernorm_X * d_attention_Q;
    common::Matrix d_weight_K = transposed_layernorm_X * d_attention_K;
    common::Matrix d_weight_V = transposed_layernorm_X * d_attention_V;
    
    common::Matrix d_layernorm_X = d_attention_Q * input.weight_Q_.transpose_copy();
    d_layernorm_X.plus(d_attention_K * input.weight_K_.transpose_copy());
    d_layernorm_X.plus(d_attention_V * input.weight_V_.transpose_copy());

    common::Matrix d_input_X;
    common::Vector d_gamma_X;
    common::Vector d_beta_X;
    d_layernorm(d_layernorm_X, intermediate.input_X_, input.gamma_X_, input.eps_X_, d_input_X, d_gamma_X, d_beta_X);
    d_input_X.plus(d_attention_AVR);
    
    auto &all_position_embeddings = input.embeddings_.mutable_all_position_embeddings();
    common::Matrix d_all_position_embeddings(all_position_embeddings.row_count(), all_position_embeddings.col_count());
    d_all_position_embeddings.unique_initialize(0.0F);
    
    for (size_t i = 0; i < d_input_X.row_count(); i++)
    {
        for (size_t j = 0; j < d_input_X.col_count(); j++)
        {
            d_all_token_embeddings.set_element(input.token_ids_[i], j, d_all_token_embeddings.get_element(input.token_ids_[i], j) + d_input_X.get_element(i, j));
            d_all_position_embeddings.set_element(input.position_indices_[i], j, d_all_position_embeddings.get_element(input.position_indices_[i], j) + d_input_X.get_element(i, j));
        }
    }

    if (input_gradients != nullptr)
    {
        input_gradients->d_all_token_embeddings_    = d_all_token_embeddings;
        input_gradients->d_all_position_embeddings_ = d_all_position_embeddings;

        input_gradients->d_gamma_X_ = d_gamma_X;
        input_gradients->d_beta_X_  = d_beta_X;

        input_gradients->d_weight_Q_ = d_weight_Q;
        input_gradients->d_weight_K_ = d_weight_K;
        input_gradients->d_weight_V_ = d_weight_V;

        input_gradients->d_gamma_A_ = d_gamma_A;
        input_gradients->d_beta_A_  = d_beta_A;

        input_gradients->d_up_projector_   = d_up_projector;
        input_gradients->d_down_projector_ = d_down_projector;
        input_gradients->d_up_bias_        = d_up_bias;
        input_gradients->d_down_bias_      = d_down_bias;
    }

    if (!update || util::equal(learning_rate, 0.0F) || learning_rate < 0.0F)
    {
        return;
    }

#define BACKWARD_UPDATE(parameter_name) \
    d_##parameter_name.scale(- learning_rate); \
    input.parameter_name##_.plus(d_##parameter_name);

    BACKWARD_UPDATE(all_token_embeddings)
    BACKWARD_UPDATE(all_position_embeddings)

    BACKWARD_UPDATE(gamma_X)
    BACKWARD_UPDATE(beta_X)

    BACKWARD_UPDATE(weight_Q)
    BACKWARD_UPDATE(weight_K)
    BACKWARD_UPDATE(weight_V)

    BACKWARD_UPDATE(gamma_A)
    BACKWARD_UPDATE(beta_A)

    BACKWARD_UPDATE(up_projector)
    BACKWARD_UPDATE(down_projector)
    BACKWARD_UPDATE(up_bias)
    BACKWARD_UPDATE(down_bias)

#undef BACKWARD_UPDATE
}

}
