#pragma once

#include "shared.h"

namespace transformer
{

float d_gelu_scalar(float x);
float d_gelu_tanh_input_scalar(float x);

void d_softmax(const common::Matrix &result, const common::Matrix &d_result, common::Matrix &d_input);
void d_layernorm(const common::Matrix &d_result, const common::Matrix &input, const common::Vector &gamma, float eps,
        common::Matrix &d_input, common::Vector &d_gamma, common::Vector &d_beta);

struct Input_Gradients
{
    common::Matrix d_all_token_embeddings_;
    common::Matrix d_all_position_embeddings_;
    
    common::Vector d_gamma_X_;
    common::Vector d_beta_X_;
    
    common::Matrix d_weight_Q_;
    common::Matrix d_weight_K_;
    common::Matrix d_weight_V_;
    
    common::Vector d_gamma_A_;
    common::Vector d_beta_A_;
    
    common::Matrix d_up_projector_;
    common::Matrix d_down_projector_;
    common::Vector d_up_bias_;
    common::Vector d_down_bias_;
};

void backward(Input &input, const Result &result, const Intermediate &intermediate,
        Input_Gradients *input_gradients = nullptr,
        bool update = false, float learning_rate = 0.0F);

}
