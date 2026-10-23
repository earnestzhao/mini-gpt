#pragma once

#include <string>
#include <vector>
#include <cstddef>

namespace tokenizer
{

class Token
{
public:
    Token();
    Token(std::size_t id, std::string value);

public:
    bool is_word();
    bool is_special();
    bool is_unknown();
    bool is_invalid();

public:
    std::size_t id_;
    std::string value_;

public:
    static bool is_word_token_id(std::size_t token_id);
    static bool is_unknown_token_id(std::size_t token_id);
    static bool is_invalid_token_id(std::size_t token_id);
    static bool is_special_token_id(std::size_t token_id);
    static bool is_special_token_string(const std::string &token);

public:
    static bool explicit_defined_splitter(char c);
    static bool can_character_be_token(char c);
    static bool can_be_merged(const std::pair<std::string, std::string> &token_pair);
};

class Trainer
{
public:
    virtual int train(const std::string &input, std::vector<std::string> &tokens) const = 0;
};

class Encoder
{
public:
    virtual int to_tokens(const std::string &input, std::vector<std::string> &tokens) const = 0;
    virtual int to_token_ids(const std::string &input, std::vector<std::size_t> &token_ids) const;
    virtual int to_token_objects(const std::string &input, std::vector<Token> &token_objects) const;
    virtual int convert_tokens_to_ids(std::vector<std::string> &tokens, std::vector<std::size_t> &token_ids) const = 0;
};

}
