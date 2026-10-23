#include "bpe_trainer.h"
#include "greedy_encoder.h"

#include <set>
#include <map>
#include <algorithm>
#include <unordered_map>

namespace tokenizer
{

BPE_Trainer::BPE_Trainer()
{

}

BPE_Trainer::BPE_Trainer(std::size_t max_scanning_times, std::size_t target_token_count)
        : max_scanning_times_(max_scanning_times), target_token_count_(target_token_count)
{

}

int BPE_Trainer::train(const std::string &input, std::vector<std::string> &tokens) const
{
    // V0 Implementation Rules:
    // 1. Each single character is a token.
    // 2. Only letters and digits can be merged.
    // 3. Letters merge with letters, digits merge with digits.

    std::vector<std::string> token_values;
    for (auto ch = ASCII_CHARACTER_MIN; ch <= ASCII_CHARACTER_MAX; ch++)
    {
        if (Token::can_character_be_token(ch))
        {
            token_values.emplace_back(std::string(1, ch));
        }

        // unsigned char needs this code to quit loop
        if (ch == ASCII_CHARACTER_MAX)
        {
            break;
        }
    }
    
    Greedy_Encoder helper_encoder;
    size_t current_running_round = 0;
    while (current_running_round ++ < max_scanning_times_ && token_values.size() < target_token_count_)
    {
        int retcode = helper_encoder.load_tokens_without_ids(token_values);
        if (retcode < 0)
        {
            return -1;
        }

        std::vector<std::string> input_tokens;
        retcode = helper_encoder.to_tokens(input, input_tokens);
        if (retcode < 0)
        {
            return -2;
        }

        if (input_tokens.empty())
        {
            return -3;
        }

        struct Pair_String_Hash
        {
            std::size_t operator()(const std::pair<std::string, std::string> &key) const
            {
                std::size_t hash_first  = std::hash<std::string>{}(key.first);
                std::size_t hash_second = std::hash<std::string>{}(key.second);
                return hash_first ^ (hash_second + 0x9e3779b9 + (hash_first << 6) + (hash_first >> 2));
            }
        };
        
        std::unordered_map<std::pair<std::string, std::string>, size_t, Pair_String_Hash> adjacent_token_pair_count;
        std::map<size_t, std::set<std::pair<std::string, std::string>>, std::greater<size_t>> count_to_adjacent_token_pair_set;
        for (size_t i = 0; i < input_tokens.size() - 1; i++)
        {
            auto adjacent_token_pair = std::make_pair(input_tokens[i], input_tokens[i + 1]);
            auto iterator = adjacent_token_pair_count.find(adjacent_token_pair);
            if (iterator != adjacent_token_pair_count.end())
            {
                count_to_adjacent_token_pair_set[iterator->second].erase(adjacent_token_pair);
                count_to_adjacent_token_pair_set[++ iterator->second].insert(adjacent_token_pair);
            }
            else
            {
                adjacent_token_pair_count[adjacent_token_pair] = 1;
                count_to_adjacent_token_pair_set[1].insert(adjacent_token_pair);
            }
        }
        
        size_t updated_count = 0;
        size_t need_to_update_number = target_token_count_ / max_scanning_times_;
        if (need_to_update_number == 0)
        {
            need_to_update_number = 1;
        }

        for (const auto &count_to_token_pair_set_item : count_to_adjacent_token_pair_set)
        {
            for (const auto &token_pair : count_to_token_pair_set_item.second)
            {
                if (Token::can_be_merged(token_pair))
                {
                    auto new_token = token_pair.first + token_pair.second;
                    token_values.push_back(new_token);
                    updated_count ++;
                }
            }

            if (updated_count >= need_to_update_number)
            {
                break;
            }
        }
    }
    
    std::sort(token_values.begin(), token_values.end());
    tokens.swap(token_values);
    return 0;
}

}
