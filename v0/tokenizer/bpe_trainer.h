#pragma once

#include "base.h"

namespace tokenizer
{

class BPE_Trainer : public Trainer
{
public:
    BPE_Trainer();
    BPE_Trainer(std::size_t max_scanning_times, std::size_t target_token_count);

public:
    int train(const std::string &input, std::vector<std::string> &tokens) const override;

private:
    size_t max_scanning_times_ = 1;
    size_t target_token_count_ = 1000;
};

}
