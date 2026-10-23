#include "embeddings.h"

using namespace common;
using namespace embedding;

////////////////////

Embeddings::Embeddings(size_t max_token_id, size_t max_sequence_length, size_t embedding_dim_count)
        : all_token_embeddings_(max_token_id + 1, embedding_dim_count),
          all_position_embeddings_(max_sequence_length, embedding_dim_count)
{

}

////////////////////

void Embeddings::random_initialize(
        float token_embeddings_random_range_start, float token_embeddings_random_range_end,
        float position_embeddings_random_range_start, float position_embeddings_random_range_end)
{
    all_token_embeddings_.random_initialize(token_embeddings_random_range_start, token_embeddings_random_range_end);
    all_position_embeddings_.random_initialize(position_embeddings_random_range_start, position_embeddings_random_range_end);
}

////////////////////

std::string Embeddings::serialize() const
{
    std::string serialized;
    serialized += all_token_embeddings_.serialize();
    serialized += SERIALIZING_SPLITTER;
    serialized += all_position_embeddings_.serialize();
    return serialized;
}

Embeddings::EMBEDDINGS_DESERIALIZE_ERROR Embeddings::deserialize(const std::string &input)
{
    auto splitter_position = input.find(SERIALIZING_SPLITTER);
    if (splitter_position == std::string::npos)
    {
        return EDE_INVALID_INPUT;
    }
    
    auto serialized_all_token_embeddings = input.substr(0, splitter_position);
    auto serialized_all_position_embeddings = input.substr(splitter_position + SERIALIZING_SPLITTER.length());
    
    auto retcode = all_token_embeddings_.deserialize(serialized_all_token_embeddings);
    if (retcode != Matrix::MDE_OK)
    {
        return EDE_TOKEN_EMBEDDINGS_DESERIALIZE_ERROR;
    }

    retcode = all_position_embeddings_.deserialize(serialized_all_position_embeddings);
    if (retcode != Matrix::MDE_OK)
    {
        return EDE_POSITION_EMBEDDINGS_DESERIALIZE_ERROR;
    }

    return EDE_OK;
}

////////////////////

Vector_View Embeddings::token_embedding(size_t token_id)
{
    return get_embedding(all_token_embeddings_, token_id);
}

Vector_View Embeddings::position_embedding(size_t position)
{
    return get_embedding(all_position_embeddings_, position);
}

Matrix Embeddings::token_embeddings(const std::vector<size_t> &token_ids)
{
    Matrix result;
    get_embeddings(all_token_embeddings_, token_ids, result);
    return result;
}

Matrix Embeddings::position_embeddings(const std::vector<size_t> &positions)
{
    Matrix result;
    get_embeddings(all_position_embeddings_, positions, result);
    return result;
}

////////////////////

const Matrix &Embeddings::all_token_embeddings()
{
    return all_token_embeddings_;
}

const Matrix &Embeddings::all_position_embeddings()
{
    return all_position_embeddings_;
}

Matrix &Embeddings::mutable_all_token_embeddings()
{
    return all_token_embeddings_;
}

Matrix &Embeddings::mutable_all_position_embeddings()
{
    return all_position_embeddings_;
}

////////////////////

Vector_View Embeddings::get_embedding(const common::Matrix &all_embeddings, size_t index)
{
    return all_embeddings.row_vector(index);
}

void Embeddings::get_embeddings(const Matrix &all_embeddings, const std::vector<size_t> &indexes, Matrix &embeddings)
{
    embeddings.resize(indexes.size(), all_embeddings.col_count());
    
    for (size_t i = 0; i < indexes.size(); i++)
    {
        get_embedding(all_embeddings, indexes[i]).copy_to(embeddings.row_vector(i));
    }
}
