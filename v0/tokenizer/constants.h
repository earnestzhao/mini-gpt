#pragma once

#include <string>
#include <cstddef>

namespace tokenizer
{

enum SPECIAL_TOKEN_ID
{
    STID_UNKNOWN = 0,
    STID_INVALID = 1,
    
    STID_USR_TEXT_START = 10,
    STID_USR_TEXT_END   = 11,
    STID_GEN_TEXT_START = 12,
    STID_GEN_TEXT_END   = 13,
    
    STID_WORKING    = 100,
    STID_THINKING   = 101,
    STID_SEARCHING  = 102,
    STID_USING_TOOL = 103,
    
    STID_MAX = 999,
};

const std::string SPECIAL_TOKEN_ABBR   = "ST";
const std::string SPECIAL_TOKEN_PREFIX = "||[";
const std::string SPECIAL_TOKEN_SUFFIX = "]||";

const std::string ST_UNKNOWN = SPECIAL_TOKEN_PREFIX + "UNK" + SPECIAL_TOKEN_SUFFIX;
const std::string ST_INVALID = SPECIAL_TOKEN_PREFIX + "IVT" + SPECIAL_TOKEN_SUFFIX;

const std::string ST_USR_TEXT_START = SPECIAL_TOKEN_PREFIX + "UTS" + SPECIAL_TOKEN_SUFFIX;
const std::string ST_USR_TEXT_END   = SPECIAL_TOKEN_PREFIX + "UTE" + SPECIAL_TOKEN_SUFFIX;
const std::string ST_GEN_TEXT_START = SPECIAL_TOKEN_PREFIX + "GTS" + SPECIAL_TOKEN_SUFFIX;
const std::string ST_GEN_TEXT_END   = SPECIAL_TOKEN_PREFIX + "GTE" + SPECIAL_TOKEN_SUFFIX;

const std::string ST_WORKING    = SPECIAL_TOKEN_PREFIX + "WKG" + SPECIAL_TOKEN_SUFFIX;
const std::string ST_THINKING   = SPECIAL_TOKEN_PREFIX + "TKG" + SPECIAL_TOKEN_SUFFIX;
const std::string ST_SEARCHING  = SPECIAL_TOKEN_PREFIX + "SRG" + SPECIAL_TOKEN_SUFFIX;
const std::string ST_USING_TOOL = SPECIAL_TOKEN_PREFIX + "UST" + SPECIAL_TOKEN_SUFFIX;

const std::size_t MIN_WORD_TOKEN_ID = STID_MAX + 1;
const std::size_t WORD_TOKEN_PLACEHOLDER_ID = MIN_WORD_TOKEN_ID;

const std::size_t WORD_TOKEN_BASE_ID  = MIN_WORD_TOKEN_ID + 1;
const std::size_t WORD_TOKEN_START_ID = WORD_TOKEN_BASE_ID;

const unsigned char ASCII_CHARACTER_MIN = 0;
const unsigned char ASCII_CHARACTER_MAX = 255;

const std::size_t EXPECTED_TOKEN_START_ID = STID_USR_TEXT_START;
inline bool unexpected_token_id(size_t token_id)
{
    return token_id == STID_UNKNOWN || token_id == STID_INVALID;
}

}
