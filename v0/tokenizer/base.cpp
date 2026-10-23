#include "base.h"
#include "constants.h"
#include <algorithm>

namespace tokenizer
{

////////////////////

Token::Token()
{

}

Token::Token(std::size_t id, std::string value) : id_(id), value_(value)
{

}

bool Token::is_word()
{
    return is_word_token_id(id_);
}

bool Token::is_special()
{
    return is_special_token_id(id_);
}

bool Token::is_unknown()
{
    return is_unknown_token_id(id_);
}

bool Token::is_invalid()
{
    return is_invalid_token_id(id_);
}

////////////////////

bool Token::is_word_token_id(std::size_t token_id)
{
    return token_id >= MIN_WORD_TOKEN_ID;
}

bool Token::is_unknown_token_id(std::size_t token_id)
{
    return token_id == STID_UNKNOWN;
}

bool Token::is_invalid_token_id(std::size_t token_id)
{
    return token_id == STID_INVALID;
}

bool Token::is_special_token_id(std::size_t token_id)
{
    return token_id <= STID_MAX;
}

bool Token::is_special_token_string(const std::string &token)
{
    if (token.length() <= SPECIAL_TOKEN_PREFIX.length() || token.length() <= SPECIAL_TOKEN_SUFFIX.length())
    {
        return false;
    }

    return token.substr(0, SPECIAL_TOKEN_PREFIX.length()) == SPECIAL_TOKEN_PREFIX
        && token.substr(token.length() - SPECIAL_TOKEN_SUFFIX.length()) == SPECIAL_TOKEN_SUFFIX;
}

////////////////////

bool Token::explicit_defined_splitter([[maybe_unused]] char c)
{
    // return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    return false;
}

bool Token::can_character_be_token([[maybe_unused]] char c)
{
    // return c != ' ';
    return true;
}

bool Token::can_be_merged(const std::pair<std::string, std::string> &token_pair)
{
    if (token_pair.first.empty() || token_pair.second.empty())
    {
        return true;
    }

    auto is_pure_digits = [](const std::string& token)
    {
        return !token.empty() && std::all_of(token.begin(), token.end(), [](unsigned char c) { return std::isdigit(c); });
    };

    auto is_pure_letters = [](const std::string& token)
    {
        return !token.empty() && std::all_of(token.begin(), token.end(), [](unsigned char c) { return std::isalpha(c); });
    };
    
    return (is_pure_digits(token_pair.first) && is_pure_digits(token_pair.second))
        || (is_pure_letters(token_pair.first) && is_pure_letters(token_pair.second));
}

////////////////////

int Encoder::to_token_ids(const std::string &input, std::vector<std::size_t> &token_ids) const
{
    std::vector<std::string> tokens;
    int retcode = to_tokens(input, tokens);
    if (retcode < 0)
    {
        token_ids.clear();
        return -1;
    }

    retcode = convert_tokens_to_ids(tokens, token_ids);
    if (retcode < 0)
    {
        token_ids.clear();
        return -2;
    }

    return 0;
}

int Encoder::to_token_objects(const std::string &input, std::vector<Token> &token_objects) const
{
    std::vector<std::string> tokens;
    int retcode = to_tokens(input, tokens);
    if (retcode < 0)
    {
        token_objects.clear();
        return -1;
    }
    
    std::vector<std::size_t> token_ids;
    retcode = convert_tokens_to_ids(tokens, token_ids);
    if (retcode < 0)
    {
        token_objects.clear();
        return -2;
    }

    if (tokens.size() != token_ids.size())
    {
        token_objects.clear();
        return -3;
    }

    for (std::size_t i = 0; i < tokens.size(); i++)
    {
        token_objects.emplace_back(token_ids[i], tokens[i]);
    }

    return 0;
}

}
