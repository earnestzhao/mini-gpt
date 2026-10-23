#pragma once

#include "constants.h"
#include "common/matrix.h"

#include <vector>
#include <string>

namespace embedding
{

class Embeddings
{
public:
    Embeddings() = delete;
    Embeddings(size_t max_token_id, size_t max_sequence_length, size_t embedding_dim_count = DEFAULT_DIM_COUNT);

public:
    void random_initialize(
            float token_embeddings_random_range_start, float token_embeddings_random_range_end,
            float position_embeddings_random_range_start, float position_embeddings_random_range_end);

public:
    std::string serialize() const;
    enum EMBEDDINGS_DESERIALIZE_ERROR
    {
        EDE_OK = 0,
        EDE_INVALID_INPUT = -1,
        EDE_TOKEN_EMBEDDINGS_DESERIALIZE_ERROR = -2,
        EDE_POSITION_EMBEDDINGS_DESERIALIZE_ERROR = -3,
    };
    EMBEDDINGS_DESERIALIZE_ERROR deserialize(const std::string &input);

public:
    common::Vector_View token_embedding(size_t token_id);
    common::Vector_View position_embedding(size_t position);

public:
    common::Matrix token_embeddings(const std::vector<size_t> &token_ids);
    common::Matrix position_embeddings(const std::vector<size_t> &positions);

public:
    const common::Matrix &all_token_embeddings();
    const common::Matrix &all_position_embeddings();
    common::Matrix &mutable_all_token_embeddings();
    common::Matrix &mutable_all_position_embeddings();

private:
    common::Vector_View get_embedding(const common::Matrix &all_embeddings, size_t index);
    void get_embeddings(const common::Matrix &all_embeddings, const std::vector<size_t> &indexes, common::Matrix &embeddings);

private:
    common::Matrix all_token_embeddings_;
    common::Matrix all_position_embeddings_;
};

}
