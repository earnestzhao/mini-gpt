#pragma once

#include "shared.h"

namespace transformer
{

common::Matrix &layernorm(common::Matrix &input_output,
        const common::Vector &gamma, const common::Vector &beta, float eps,
        bool vectors_by_row = true);

common::Matrix layernorm_copy(const common::Matrix &input,
        const common::Vector &gamma, const common::Vector &beta, float eps,
        bool vectors_by_row = true);

void layernorm_to(const common::Matrix &input,
        const common::Vector &gamma, const common::Vector &beta, float eps,
        common::Matrix &result, bool vectors_by_row = true);

common::Matrix &ffn(common::Matrix &input_output,
        const common::Matrix &up_projector, const common::Matrix &down_projector,
        const common::Vector &up_bias, const common::Vector &down_bias,
        bool vectors_by_row = true);

common::Matrix ffn_copy(const common::Matrix &input,
        const common::Matrix &up_projector, const common::Matrix &down_projector,
        const common::Vector &up_bias, const common::Vector &down_bias,
        bool vectors_by_row = true);

void ffn_to(const common::Matrix &input,
        const common::Matrix &up_projector, const common::Matrix &down_projector,
        const common::Vector &up_bias, const common::Vector &down_bias,
        common::Matrix &result, bool vectors_by_row = true);

enum Forward_Mode
{
    FM_INVALID      = 0,
    FM_TRAINING     = 1,
    FM_GENERATING   = 2,
    FM_TESTING_BOTH = FM_TRAINING | FM_GENERATING,
};

void forward(const Input &input, Result &result, Intermediate &intermediate,
        Forward_Mode mode = FM_TESTING_BOTH, bool sampling_generation = false);

}
