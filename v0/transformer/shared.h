#pragma once

#include "common/util.h"
#include "common/matrix.h"
#include "common/vector.h"
#include "embedding/embeddings.h"

#include <vector>
#include <string>

namespace transformer
{

struct Input
{
    Input(embedding::Embeddings &embeddings);

    embedding::Embeddings &embeddings_;
    common::Matrix &all_token_embeddings_;
    common::Matrix &all_position_embeddings_;

    std::vector<size_t> token_ids_;
    std::vector<size_t> position_indices_;
    
    common::Vector gamma_X_;
    common::Vector beta_X_;
    float eps_X_ = util::SMALL_EPS;
    
    common::Matrix weight_Q_;
    common::Matrix weight_K_;
    common::Matrix weight_V_;
    
    common::Vector gamma_A_;
    common::Vector beta_A_;
    float eps_A_ = util::SMALL_EPS;
    
    common::Matrix up_projector_;
    common::Matrix down_projector_;
    common::Vector up_bias_;
    common::Vector down_bias_;
    
    const std::string SERIALIZING_SPLITTER = "||||||||||-----TRANSFORMER-----||||||||||\n";
    std::string serialize_training_items() const;
    
    enum DESERIALIZE_ERROR
    {
        TIDE_OK = 0,
        TIDE_INVALID_INPUT = -1,

        TIDE_EMBEDDINGS_FAILED     = -2,
        TIDE_GAMMA_X_FAILED        = -3,
        TIDE_BETA_X_FAILED         = -4,
        TIDE_WEIGHT_Q_FAILED       = -5,
        TIDE_WEIGHT_K_FAILED       = -6,
        TIDE_WEIGHT_V_FAILED       = -7,
        TIDE_GAMMA_A_FAILED        = -8,
        TIDE_BETA_A_FAILED         = -9,
        TIDE_UP_PROJECTOR_FAILED   = -10,
        TIDE_DOWN_PROJECTOR_FAILED = -11,
        TIDE_UP_BIAS_FAILED        = -12,
        TIDE_DOWN_BIAS_FAILED      = -13,
    };
    DESERIALIZE_ERROR deserialize_training_items(const std::string &serialized);
};

struct Result
{
    common::Matrix before_joining_vocab_;
    common::Matrix after_joining_vocab_;
    common::Matrix softmax_;

    struct Loss_Item
    {
        float  value_;
        size_t on_token_id_;
    };
    
    float loss() const;
    std::vector<Loss_Item> loss_;

    struct Logit_Item
    {
        size_t candidate_token_id_;
        float  softmax_probability_;
    };
    
    size_t generated_token_id() const;
    std::vector<Logit_Item> logits(size_t index) const;
    std::vector<Logit_Item> generated_logits() const;
    std::string debugging_generated_logits() const;
    std::vector<size_t> output_token_ids_;
};

struct Intermediate
{
    common::Matrix input_token_embeddings_;
    common::Matrix input_position_embeddings_;

    common::Matrix input_X_;
    common::Matrix layernorm_X_;

    common::Matrix attention_Q_;
    common::Matrix attention_K_;
    common::Matrix attention_V_;

    common::Matrix attention_A_;
    common::Matrix attention_SA_;
    common::Matrix attention_AV_;
    common::Matrix attention_AVR_;
    
    common::Matrix layernorm_AVR_;
    common::Matrix ffn_;
    common::Matrix ffn_R_;
};

common::Matrix &casual_mask(common::Matrix &input_output, bool is_forward = true);
common::Matrix casual_mask_copy(const common::Matrix &input, bool is_forward = true);
void casual_mask_to(const common::Matrix &input, common::Matrix &result, bool is_forward = true);

common::Matrix &softmax(common::Matrix &input_output, bool vectors_by_row = true);
common::Matrix softmax_copy(const common::Matrix &input, bool vectors_by_row = true);
void softmax_to(const common::Matrix &input, common::Matrix &result, bool vectors_by_row = true);

struct Standardize_Intermediate
{
    common::Vector average_;
    common::Vector variance_;
    common::Vector denominator_;
};

common::Matrix &standardize(common::Matrix &input_output, float eps,
        Standardize_Intermediate &intermediate,
        bool vectors_by_row = true);

common::Matrix standardize_copy(const common::Matrix &input, float eps,
        Standardize_Intermediate &intermediate,
        bool vectors_by_row = true);

void standardize_to(const common::Matrix &input, float eps,
        common::Matrix &result, Standardize_Intermediate &intermediate,
        bool vectors_by_row = true);

float gelu_scalar(float x);
float gelu_tanh_input_scalar(float x);
common::Matrix &gelu(common::Matrix &input_output);
common::Matrix gelu_copy(const common::Matrix &input);
void gelu_to(const common::Matrix &input, common::Matrix &result);

common::Matrix duplicate_vector_to_matrix(const common::Vector &v, size_t dup_count, bool vectors_by_row = true);

}
