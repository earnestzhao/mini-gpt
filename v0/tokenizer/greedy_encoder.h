#pragma once

#include "base.h"
#include "constants.h"
#include <unordered_map>

namespace tokenizer
{

class Greedy_Encoder : public Encoder
{
public:
    enum TOKENIZE_ERROR
    {
        TE_FULL_SUCCESS = 0,
        TE_SUCCESS_WITH_UNKNOWN_TOKENS = 1,
        TE_SUCCESS_WITH_INVALID_TOKENS = 2,
        TE_SUCCESS_WITH_SPECIAL_TOKENS = 3,
        
        TE_EMPTY_INPUT = -1,
        TE_INVALID_INPUT = -2,
        TE_CONFLICTED_TOKEN_IDS = -3,
    };

public:
    int to_tokens(const std::string &input, std::vector<std::string> &tokens) const override;
    int convert_tokens_to_ids(std::vector<std::string> &tokens, std::vector<std::size_t> &token_ids) const override;

public:
    enum LOAD_TOKENS_OPERATION_MODE
    {
        OVERWRITE_ALL,
        FAIL_ON_CONFLICT,
        IGNORE_ON_CONFLICT,
        UPDATE_ON_CONFLICT,
    };

    int load_tokens_with_ids(
            const std::vector<std::string> &tokens,
            const std::vector<std::size_t> &token_ids,
            LOAD_TOKENS_OPERATION_MODE operation_mode = OVERWRITE_ALL);
    
    // It seems that load_tokens_without_ids should only be used by trainer.
    int load_tokens_without_ids(
            const std::vector<std::string> &tokens,
            LOAD_TOKENS_OPERATION_MODE operation_mode = OVERWRITE_ALL);

    int load_tokens_using_base_plus_index_id(
            const std::vector<std::string> &tokens,
            std::size_t token_id_base = WORD_TOKEN_BASE_ID,
            LOAD_TOKENS_OPERATION_MODE operation_mode = OVERWRITE_ALL);

    int load_tokens_by_objects(
            const std::vector<Token> &tokens,
            LOAD_TOKENS_OPERATION_MODE operation_mode = OVERWRITE_ALL);

private:
    enum TOKEN_DICTIONARY_KEY_TYPE
    {
        TOKEN_PREFIX,
        INVALID_TOKEN,
        COMPLETE_TOKEN,
    };

    // Token or token prefix -> Type of Key String
    std::unordered_map<std::string, TOKEN_DICTIONARY_KEY_TYPE> token_dictionary_;
    
    // Token -> Token ID
    std::unordered_map<std::string, std::size_t> token_id_map_;
};

}
