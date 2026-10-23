#include "greedy_encoder.h"
#include "constants.h"
#include <cassert>

namespace tokenizer
{

////////////////////

int Greedy_Encoder::to_tokens(const std::string &input, std::vector<std::string> &tokens) const
{
    if (input.empty())
    {
        return TE_EMPTY_INPUT;
    }

    tokens.clear();
    
    std::string current_identifying_token;
    size_t current_identifying_token_start_index = 0;
    size_t current_identifying_token_max_complete_sub_token_length = 0;

    bool unknown_token_detected = false;
    bool invalid_token_detected = false;
    bool special_token_detected = false;

    for (size_t i = 0; i < input.length(); i++)
    {
        if (current_identifying_token.empty() && Token::explicit_defined_splitter(input[i]))
        {
            continue;
        }
        
        current_identifying_token.push_back(input[i]);
        if (current_identifying_token.length() == 1)
        {
            current_identifying_token_start_index = i;
        }
        
        auto iterator = token_dictionary_.find(current_identifying_token);
        if (iterator == token_dictionary_.end() || i == input.length() - 1)
        {
            // Deal with the last charater from input if exists in token_dictionary_.
            if (iterator != token_dictionary_.end())
            {
                assert(iterator->second >= TOKEN_PREFIX && iterator->second <= COMPLETE_TOKEN);
                if (iterator->second != TOKEN_PREFIX)
                {
                    current_identifying_token_max_complete_sub_token_length = current_identifying_token.length();
                }
            }

            if (current_identifying_token_max_complete_sub_token_length == 0)
            {
                tokens.push_back(ST_UNKNOWN);
                unknown_token_detected = true;
                
                i = current_identifying_token_start_index;
                current_identifying_token = "";
                continue;
            }
            
            auto identified_token = current_identifying_token.substr(0, current_identifying_token_max_complete_sub_token_length);
            auto identified_token_type = token_dictionary_.at(identified_token);
            if (identified_token_type == COMPLETE_TOKEN)
            {
                tokens.push_back(identified_token);
                if (!special_token_detected)
                {
                    special_token_detected = Token::is_special_token_string(identified_token);
                }
            }
            else if (identified_token_type == INVALID_TOKEN)
            {
                tokens.push_back(ST_INVALID);
                invalid_token_detected = true;
            }
            else
            {
                assert(identified_token_type == COMPLETE_TOKEN || identified_token_type == INVALID_TOKEN);
            }
            
            i = current_identifying_token_start_index + current_identifying_token_max_complete_sub_token_length - 1;
            current_identifying_token_max_complete_sub_token_length = 0;
            current_identifying_token = "";
        }
        else
        {
            assert(iterator->second >= TOKEN_PREFIX && iterator->second <= COMPLETE_TOKEN);
            if (iterator->second != TOKEN_PREFIX)
            {
                current_identifying_token_max_complete_sub_token_length = current_identifying_token.length();
            }
        }
    }

    if (invalid_token_detected)
    {
        return TE_SUCCESS_WITH_INVALID_TOKENS;
    }

    if (unknown_token_detected)
    {
        return TE_SUCCESS_WITH_UNKNOWN_TOKENS;
    }

    if (special_token_detected)
    {
        return TE_SUCCESS_WITH_SPECIAL_TOKENS;
    }
    
    return TE_FULL_SUCCESS;
}

int Greedy_Encoder::convert_tokens_to_ids(std::vector<std::string> &tokens, std::vector<std::size_t> &token_ids) const
{
    token_ids.clear();

    bool unknown_token_detected = false;
    bool invalid_token_detected = false;
    bool special_token_detected = false;

    for (const auto &token : tokens)
    {
        auto iterator = token_id_map_.find(token);
        if (iterator != token_id_map_.end())
        {
            if (Token::is_invalid_token_id(iterator->second))
            {
                auto dict_iterator = token_dictionary_.find(token);
                assert(token == ST_INVALID || (dict_iterator != token_dictionary_.end() && dict_iterator->second == INVALID_TOKEN));
                invalid_token_detected = true;
            }

            if (Token::is_unknown_token_id(iterator->second))
            {
                assert(token == ST_UNKNOWN);
                unknown_token_detected = true;
            }

            if (Token::is_special_token_id(iterator->second))
            {
                assert(Token::is_special_token_string(token));
                special_token_detected = true;
            }
            
            token_ids.push_back(iterator->second);
        }
        else
        {
            token_ids.push_back(STID_UNKNOWN);
            unknown_token_detected = true;
        }
    }

    if (invalid_token_detected)
    {
        return TE_SUCCESS_WITH_INVALID_TOKENS;
    }

    if (unknown_token_detected)
    {
        return TE_SUCCESS_WITH_UNKNOWN_TOKENS;
    }

    if (special_token_detected)
    {
        return TE_SUCCESS_WITH_SPECIAL_TOKENS;
    }
    
    return TE_FULL_SUCCESS;
}

////////////////////

int Greedy_Encoder::load_tokens_with_ids(
        const std::vector<std::string> &tokens,
        const std::vector<std::size_t> &token_ids,
        LOAD_TOKENS_OPERATION_MODE operation_mode)
{
    assert(operation_mode >= OVERWRITE_ALL && operation_mode <= UPDATE_ON_CONFLICT);

    if (tokens.empty() || token_ids.empty())
    {
        return TE_EMPTY_INPUT;
    }

    if (tokens.size() != token_ids.size())
    {
        return TE_INVALID_INPUT;
    }

    bool unknown_token_detected = false;
    bool invalid_token_detected = false;
    bool special_token_detected = false;
    
    std::unordered_map<std::string, TOKEN_DICTIONARY_KEY_TYPE> new_token_dictionary;
    std::unordered_map<std::string, std::size_t> new_token_id_map;
    if (operation_mode != OVERWRITE_ALL)
    {
        new_token_dictionary = token_dictionary_;
        new_token_id_map = token_id_map_;
    }

    for (std::size_t i = 0; i < tokens.size(); i++)
    {
        auto token_id = token_ids[i];
        const auto &token = tokens[i];
        if (token.empty())
        {
            continue;
        }

        auto id_map_iterator = new_token_id_map.find(token);
        if (id_map_iterator != new_token_id_map.end())
        {
            if (id_map_iterator->second != token_id)
            {
                if (operation_mode == FAIL_ON_CONFLICT)
                {
                    return TE_CONFLICTED_TOKEN_IDS;
                }

                if (operation_mode == IGNORE_ON_CONFLICT)
                {
                    continue;
                }
            }
        }

        std::string current_sub_token;
        for (size_t j = 0; j < token.length() - 1; j++)
        {
            current_sub_token.push_back(token[j]);
            auto dict_iterator = new_token_dictionary.find(current_sub_token);
            if (dict_iterator == new_token_dictionary.end())
            {
                new_token_dictionary[current_sub_token] = TOKEN_PREFIX;
            }
        }
        
        if (Token::is_invalid_token_id(token_id))
        {
            new_token_dictionary[token] = INVALID_TOKEN;
            invalid_token_detected = true;
        }
        else
        {
            if (Token::is_unknown_token_id(token_id))
            {
                assert(token == ST_UNKNOWN);
                unknown_token_detected = true;
            }

            if (Token::is_special_token_id(token_id))
            {
                assert(Token::is_special_token_string(token));
                special_token_detected = true;
            }
            
            new_token_dictionary[token] = COMPLETE_TOKEN;
        }
        
        new_token_id_map[token] = token_id;
    }

    token_dictionary_.swap(new_token_dictionary);
    token_id_map_.swap(new_token_id_map);

    if (invalid_token_detected)
    {
        return TE_SUCCESS_WITH_INVALID_TOKENS;
    }

    if (unknown_token_detected)
    {
        return TE_SUCCESS_WITH_UNKNOWN_TOKENS;
    }

    if (special_token_detected)
    {
        return TE_SUCCESS_WITH_SPECIAL_TOKENS;
    }
    
    return TE_FULL_SUCCESS;
}

int Greedy_Encoder::load_tokens_without_ids(const std::vector<std::string> &tokens, LOAD_TOKENS_OPERATION_MODE operation_mode)
{
    assert(operation_mode != UPDATE_ON_CONFLICT);

    std::vector<std::size_t> token_ids;
    for (size_t i = 0; i < tokens.size(); i++)
    {
        token_ids.push_back(WORD_TOKEN_PLACEHOLDER_ID);
    }
    
    return load_tokens_with_ids(tokens, token_ids, operation_mode);
}

int Greedy_Encoder::load_tokens_using_base_plus_index_id(
        const std::vector<std::string> &tokens,
        std::size_t token_id_base,
        LOAD_TOKENS_OPERATION_MODE operation_mode)
{
    std::vector<std::size_t> token_ids;
    for (size_t i = 0; i < tokens.size(); i++)
    {
        token_ids.push_back(token_id_base + i);
    }

    return load_tokens_with_ids(tokens, token_ids, operation_mode);
}

int Greedy_Encoder::load_tokens_by_objects(const std::vector<Token> &tokens, LOAD_TOKENS_OPERATION_MODE operation_mode)
{
    std::vector<std::string> token_values;
    for (size_t i = 0; i < tokens.size(); i++)
    {
        token_values.push_back(tokens[i].value_);
    }

    std::vector<std::size_t> token_ids;
    for (size_t i = 0; i < tokens.size(); i++)
    {
        token_ids.push_back(tokens[i].id_);
    }
    
    return load_tokens_with_ids(token_values, token_ids, operation_mode);
}

}
