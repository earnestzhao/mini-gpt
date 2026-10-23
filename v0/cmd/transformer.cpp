#include <iostream>
#include <iomanip>

#include <string>
#include <sstream>

#include <cassert>
#include <unistd.h>

#include "tokenizer/greedy_encoder.h"
#include "embedding/embeddings.h"
#include "transformer/forward.h"
#include "transformer/backward.h"
#include "common/util.h"

using namespace std;
using namespace util;
using namespace common;
using namespace tokenizer;
using namespace embedding;
using namespace transformer;

////////////////////

struct Generated_Token
{
    size_t id_;
    string token_;
};

Generated_Token get_generated_token(const Result &result, const vector<string> &trained_tokens, const vector<string> &special_tokens)
{
    Generated_Token generated_token;
    generated_token.id_ = result.generated_token_id();
    if (generated_token.id_ == WORD_TOKEN_PLACEHOLDER_ID)
    {
        generated_token.id_ = STID_INVALID;
    }

    if (generated_token.id_ >= WORD_TOKEN_BASE_ID)
    {
        generated_token.token_ = trained_tokens[generated_token.id_ - WORD_TOKEN_BASE_ID];
        return generated_token;
    }
    
    generated_token.token_ = special_tokens[generated_token.id_];
    return generated_token;
}

////////////////////

int main(int argc, char *argv[])
{
    int opt = 0;
    string mode = "";

    bool debugging = false;
    bool sampling_generation = false;
    
    size_t training_times = 1;
    float learning_rate = 0.0F;

    string training_corpus_path_filename = "";
    string trained_tokens_path_filename = "";
    
    bool override_input_parameters = false;
    string input_parameters_path_filename = "";
    string output_parameters_path_filename = "";

    char *dummy;
    while ((opt = getopt(argc, argv, "m:c:t:n:l:i:o:rds")) != -1)
    {
        switch (opt)
        {
            case 'm':
                mode = optarg;
                break;
            case 'c':
                training_corpus_path_filename = optarg;
                break;
            case 't':
                trained_tokens_path_filename = optarg;
                break;
            case 'n':
                training_times = static_cast<size_t>(strtoul(optarg, &dummy, 10));
                break;
            case 'l':
                learning_rate = static_cast<float>(stod(optarg));
                break;
            case 'i':
                input_parameters_path_filename = optarg;
                break;
            case 'o':
                output_parameters_path_filename = optarg;
                break;
            case 'r':
                override_input_parameters = true;
                break;
            case 'd':
                debugging = true;
                break;
            case 's':
                sampling_generation = true;
                break;
            case '?':
            default:
                cerr << "Invalid Parameters! Should be \"m:c:t:n:l:i:o:rds\"" << endl;
                return -1;
        }
    }
    
    assert(mode == "training" || mode == "testing" || mode == "generating");
    Forward_Mode forward_mode = FM_TRAINING;
    if (mode == "testing")
    {
        forward_mode = FM_TESTING_BOTH;
    }
    else if (mode == "generating")
    {
        forward_mode = FM_GENERATING;
    }
    else
    {
        forward_mode = FM_TRAINING;
    }

    assert(!trained_tokens_path_filename.empty());
    switch (forward_mode)
    {
        case FM_TESTING_BOTH:
            assert(!training_corpus_path_filename.empty());
            break;
        case FM_TRAINING:
            assert(training_times > 0);
            assert(!training_corpus_path_filename.empty());
            assert((override_input_parameters && output_parameters_path_filename.empty())
                || (!override_input_parameters && !output_parameters_path_filename.empty()));
            break;
        case FM_GENERATING:
            assert(!input_parameters_path_filename.empty());
            break;
        default:
            assert(false);
            break;
    }
    
    string trained_tokens_str = "";
    int retcode = read_string_from_file(trained_tokens_path_filename, trained_tokens_str);
    if (retcode < 0)
    {
        cerr << "read_string_from_file [" << trained_tokens_path_filename << "] Failed [" << retcode << "]." << endl;
        return -2;
    }

    vector<string> trained_tokens;
    for (size_t index = 0; index != string::npos && index < trained_tokens_str.size();)
    {
        auto token_end_pos = trained_tokens_str.find("\n", index);
        trained_tokens.push_back(trained_tokens_str.substr(index, token_end_pos - index));
        index = token_end_pos + 1;
    }
    
    vector<string> special_tokens;
    for (size_t i = STID_UNKNOWN; i <= STID_MAX; i++)
    {
        ostringstream oss;
        oss << i;
        special_tokens.push_back(SPECIAL_TOKEN_PREFIX + SPECIAL_TOKEN_ABBR + oss.str() + SPECIAL_TOKEN_SUFFIX);
    }

    special_tokens[STID_UNKNOWN] = ST_UNKNOWN;
    special_tokens[STID_INVALID] = ST_INVALID;

    special_tokens[STID_USR_TEXT_START] = ST_USR_TEXT_START;
    special_tokens[STID_USR_TEXT_END]   = ST_USR_TEXT_END;
    special_tokens[STID_GEN_TEXT_START] = ST_GEN_TEXT_START;
    special_tokens[STID_GEN_TEXT_END]   = ST_GEN_TEXT_END;

    special_tokens[STID_WORKING]    = ST_WORKING;
    special_tokens[STID_THINKING]   = ST_THINKING;
    special_tokens[STID_SEARCHING]  = ST_SEARCHING;
    special_tokens[STID_USING_TOOL] = ST_USING_TOOL;

    Greedy_Encoder encoder;
    retcode = encoder.load_tokens_using_base_plus_index_id(special_tokens, STID_UNKNOWN);
    if (retcode < 0)
    {
        cerr << "encoder.load_tokens_using_base_plus_index_id (special_tokens) Failed [" << retcode << "]." << endl;
        return -3;
    }
    
    retcode = encoder.load_tokens_using_base_plus_index_id(trained_tokens, WORD_TOKEN_BASE_ID, Greedy_Encoder::FAIL_ON_CONFLICT);
    if (retcode < 0)
    {
        cerr << "encoder.load_tokens_using_base_plus_index_id (trained_tokens) Failed [" << retcode << "]." << endl;
        return -4;
    }
    
    string input_parameters = "";
    if (!input_parameters_path_filename.empty())
    {
        retcode = read_string_from_file(input_parameters_path_filename, input_parameters);
        if (retcode < 0)
        {
            cerr << "read_string_from_file [" << input_parameters_path_filename << "] Failed [" << retcode << "]." << endl;
            return -5;
        }
    }
    
    const size_t MAX_SEQUENCE_LENGTH      = 5000;
    const size_t EMBEDDING_DIM_COUNT      = 16;
    const size_t ATTENTION_DIM_COUNT      = 32;
    const size_t EMBEDDING_HIGH_DIM_COUNT = 64;

    Embeddings embeddings(WORD_TOKEN_BASE_ID + trained_tokens.size() - 1, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM_COUNT);
    Input transformer_input(embeddings);

    if (!input_parameters.empty())
    {
        retcode = (int)transformer_input.deserialize_training_items(input_parameters);
        if (retcode < 0)
        {
            cerr << "transformer_input.deserialize_training_items Failed [" << retcode << "]." << endl;
            return -6;
        }
    }
    else
    {
        const float WIR_LOWER  = -0.2F;
        const float WIR_HIGHER = 0.2F;
        transformer_input.embeddings_.random_initialize(WIR_LOWER, WIR_HIGHER, WIR_LOWER, WIR_HIGHER);

        transformer_input.gamma_X_.resize(EMBEDDING_DIM_COUNT);
        transformer_input.gamma_X_.random_initialize(WIR_LOWER, WIR_HIGHER);

        transformer_input.beta_X_.resize(EMBEDDING_DIM_COUNT);
        transformer_input.beta_X_.random_initialize(WIR_LOWER, WIR_HIGHER);
        
        transformer_input.weight_Q_.resize(EMBEDDING_DIM_COUNT, ATTENTION_DIM_COUNT);
        transformer_input.weight_Q_.random_initialize(WIR_LOWER, WIR_HIGHER);

        transformer_input.weight_K_.resize(EMBEDDING_DIM_COUNT, ATTENTION_DIM_COUNT);
        transformer_input.weight_K_.random_initialize(WIR_LOWER, WIR_HIGHER);

        transformer_input.weight_V_.resize(EMBEDDING_DIM_COUNT, EMBEDDING_DIM_COUNT);
        transformer_input.weight_V_.random_initialize(WIR_LOWER, WIR_HIGHER);

        transformer_input.gamma_A_.resize(EMBEDDING_DIM_COUNT);
        transformer_input.gamma_A_.random_initialize(WIR_LOWER, WIR_HIGHER);

        transformer_input.beta_A_.resize(EMBEDDING_DIM_COUNT);
        transformer_input.beta_A_.random_initialize(WIR_LOWER, WIR_HIGHER);

        transformer_input.up_projector_.resize(EMBEDDING_DIM_COUNT, EMBEDDING_HIGH_DIM_COUNT);
        transformer_input.up_projector_.random_initialize(WIR_LOWER, WIR_HIGHER);

        transformer_input.down_projector_.resize(EMBEDDING_HIGH_DIM_COUNT, EMBEDDING_DIM_COUNT);
        transformer_input.down_projector_.random_initialize(WIR_LOWER, WIR_HIGHER);

        transformer_input.up_bias_.resize(EMBEDDING_HIGH_DIM_COUNT);
        transformer_input.up_bias_.random_initialize(WIR_LOWER, WIR_HIGHER);

        transformer_input.down_bias_.resize(EMBEDDING_DIM_COUNT);
        transformer_input.down_bias_.random_initialize(WIR_LOWER, WIR_HIGHER);
    }

    vector<string> tokens;
    vector<size_t> token_ids;
    vector<size_t> position_indices;

    if (forward_mode == FM_GENERATING)
    {
        while (true)
        {
            tokens.push_back(ST_USR_TEXT_START);
            token_ids.push_back(STID_USR_TEXT_START);

            string input_from_standard;
            getline(cin, input_from_standard);
            assert(!input_from_standard.empty());

            if (input_from_standard == "quit")
            {
                break;
            }

            vector<string> tokens_from_standard;
            retcode = encoder.to_tokens(input_from_standard, tokens_from_standard);
            if (retcode < 0)
            {
                cerr << "encoder.to_tokens Failed [" << retcode << "]." << endl;
                return -10;
            }

            vector<size_t> token_ids_from_standard;
            retcode = encoder.convert_tokens_to_ids(tokens_from_standard, token_ids_from_standard);
            if (retcode < 0)
            {
                cerr << "encoder.convert_tokens_to_ids Failed [" << retcode << "]." << endl;
                return -11;
            }

            tokens_from_standard.push_back(ST_USR_TEXT_END);
            token_ids_from_standard.push_back(STID_USR_TEXT_END);
            
            tokens.insert(tokens.end(), tokens_from_standard.begin(), tokens_from_standard.end());
            token_ids.insert(token_ids.end(), token_ids_from_standard.begin(), token_ids_from_standard.end());
            
            position_indices.clear();
            for (size_t i = 0; i < token_ids.size(); i++)
            {
                position_indices.push_back(i);
            }
            
            if (position_indices.size() > MAX_SEQUENCE_LENGTH)
            {
                cerr << "position_indices size [" << position_indices.size() << "] exceeds MAX_SEQUENCE_LENGTH." << endl;
                return -12;
            }
            
            transformer_input.token_ids_ = token_ids;
            transformer_input.position_indices_ = position_indices;
            
            size_t last_generated_token_id = STID_UNKNOWN;
            const size_t MAX_ONCE_GENERATED_TOKEN_COUNT = 80;
            for (size_t i = 1; i <= MAX_ONCE_GENERATED_TOKEN_COUNT; i++)
            {
                Result forward_result;
                Intermediate intermediate;
                forward(transformer_input, forward_result, intermediate, FM_GENERATING, sampling_generation);

                auto generated_token = get_generated_token(forward_result, trained_tokens, special_tokens);
                transformer_input.token_ids_.push_back(generated_token.id_);
                transformer_input.position_indices_.push_back(transformer_input.position_indices_.size());
                if (transformer_input.position_indices_.size() > MAX_SEQUENCE_LENGTH)
                {
                    cerr << "transformer_input.position_indices_ size [" << transformer_input.position_indices_.size() << "] exceeds MAX_SEQUENCE_LENGTH." << endl;
                    return -13;
                }

                tokens.push_back(generated_token.token_);
                token_ids.push_back(generated_token.id_);
                last_generated_token_id = generated_token.id_;
                
                // V0: output all tokens including special ones.
                // If we need to have better user experience, we need to add is_special_token_id judgement here.
                if (debugging)
                {
                    cout << generated_token.token_ << "|->(" << generated_token.id_ << ")" << endl;
                    cout << forward_result.debugging_generated_logits() << endl << endl;
                }
                else
                {
                    cout << generated_token.token_;
                }
                
                if (generated_token.id_ == STID_GEN_TEXT_END)
                {
                    cout << endl << endl;
                    break;
                }

                // V0: quit if invalid/unknown token is generated.
                // Maybe we can output better words such as "Didn't generate content because some issues occur."
                if (unexpected_token_id(generated_token.id_))
                {
                    cout << "Error: Invalid/Unknown/Other Unexpected Token Generated. Need to Quit." << endl;
                    return -14;
                }
            }

            if (last_generated_token_id != STID_GEN_TEXT_END)
            {
                tokens.push_back(ST_GEN_TEXT_END);
                token_ids.push_back(STID_GEN_TEXT_END);
                cout << endl << endl;
            }
        }

        return 0;
    }

    string training_corpus;
    retcode = read_string_from_file(training_corpus_path_filename, training_corpus);
    if (retcode < 0)
    {
        cerr << "read_string_from_file [" << training_corpus_path_filename << "] Failed [" << retcode << "]." << endl;
        return -20;
    }

    retcode = encoder.to_tokens(training_corpus, tokens);
    if (retcode < 0)
    {
        cerr << "encoder.to_tokens Failed [" << retcode << "]." << endl;
        return -21;
    }

    retcode = encoder.convert_tokens_to_ids(tokens, token_ids);
    if (retcode < 0)
    {
        cerr << "encoder.convert_tokens_to_ids Failed [" << retcode << "]." << endl;
        return -22;
    }
    
    for (size_t i = 0; i < token_ids.size(); i++)
    {
        position_indices.push_back(i);
    }
    
    transformer_input.token_ids_ = token_ids;
    transformer_input.position_indices_ = position_indices;
    
    Result forward_result;
    Intermediate intermediate;
    forward(transformer_input, forward_result, intermediate, forward_mode);

    if (forward_mode == FM_TRAINING)
    {
        for (size_t i = 1; i <= training_times; i++)
        {
            cout << endl;
            cout << "Training Round: " << i << endl;
            cout << "Loss: " << forward_result.loss() << endl;
            cout << endl;
            
            backward(transformer_input, forward_result, intermediate, nullptr, true, learning_rate);
            forward(transformer_input, forward_result, intermediate, FM_TRAINING);
        }

        string output_path_filename = output_parameters_path_filename;
        if (override_input_parameters)
        {
            output_path_filename = input_parameters_path_filename;
        }
        
        retcode = dump_string_to_file(transformer_input.serialize_training_items(), output_path_filename);
        if (retcode < 0)
        {
            cerr << "dump_string_to_file [" << output_path_filename << "] Failed [" << retcode << "]." << endl;
            return -30;
        }
        
        return 0;
    }

    auto generated_token = get_generated_token(forward_result, trained_tokens, special_tokens);
    cout << endl;
    cout << "Output: " << endl;
    cout << "    Generated Token ID: " << generated_token.id_ << endl;
    cout << "    Generated Token: " << generated_token.token_ << endl;
    cout << endl << endl;

    Input_Gradients input_gradients;
    backward(transformer_input, forward_result, intermediate, &input_gradients);
    const float TESTING_EPS = 1e-2;

#define TEST_GRADIENT_MI(parameter_name, row_index, col_index) \
    { \
        auto element_value = transformer_input.parameter_name##_.get_element(row_index, col_index); \
        transformer_input.parameter_name##_.set_element(row_index, col_index, element_value + TESTING_EPS); \
        forward(transformer_input, forward_result, intermediate, FM_TRAINING); \
        auto loss_plus_eps = forward_result.loss(); \
        \
        transformer_input.parameter_name##_.set_element(row_index, col_index, element_value - TESTING_EPS); \
        forward(transformer_input, forward_result, intermediate, FM_TRAINING); \
        auto loss_minus_eps = forward_result.loss(); \
        transformer_input.parameter_name##_.set_element(row_index, col_index, element_value); \
        \
        auto testing_standard = (loss_plus_eps - loss_minus_eps) / (2 * TESTING_EPS); \
        auto testing_gradient = input_gradients.d_##parameter_name##_.get_element(row_index, col_index); \
        \
        cout << "Testing Element: "       << #parameter_name << "[" << row_index << "][" << col_index << "]" << endl; \
        cout << "Testing Element Value: " << setprecision(9) << element_value                                << endl; \
        cout << "Testing Loss(+eps): "    << setprecision(9) << loss_plus_eps                                << endl; \
        cout << "Testing Loss(-eps): "    << setprecision(9) << loss_minus_eps                               << endl; \
        cout << "Testing Standard: "      << setprecision(9) << testing_standard                             << endl; \
        cout << "Testing Gradient: "      << setprecision(9) << testing_gradient                             << endl; \
        cout << "Testing Difference: "    << setprecision(9) << testing_standard - testing_gradient          << endl; \
        cout << endl; \
    }

#define TEST_GRADIENT_VI(parameter_name, index) \
    { \
        auto element_value = transformer_input.parameter_name##_.get_element(index); \
        transformer_input.parameter_name##_.set_element(index, element_value + TESTING_EPS); \
        forward(transformer_input, forward_result, intermediate, FM_TRAINING); \
        auto loss_plus_eps = forward_result.loss(); \
        \
        transformer_input.parameter_name##_.set_element(index, element_value - TESTING_EPS); \
        forward(transformer_input, forward_result, intermediate, FM_TRAINING); \
        auto loss_minus_eps = forward_result.loss(); \
        transformer_input.parameter_name##_.set_element(index, element_value); \
        \
        auto testing_standard = (loss_plus_eps - loss_minus_eps) / (2 * TESTING_EPS); \
        auto testing_gradient = input_gradients.d_##parameter_name##_.get_element(index); \
        \
        cout << "Testing Element: "       << #parameter_name << "[" << index << "]"                 << endl; \
        cout << "Testing Element Value: " << setprecision(9) << element_value                       << endl; \
        cout << "Testing Loss(+eps): "    << setprecision(9) << loss_plus_eps                       << endl; \
        cout << "Testing Loss(-eps): "    << setprecision(9) << loss_minus_eps                      << endl; \
        cout << "Testing Standard: "      << setprecision(9) << testing_standard                    << endl; \
        cout << "Testing Gradient: "      << setprecision(9) << testing_gradient                    << endl; \
        cout << "Testing Difference: "    << setprecision(9) << testing_standard - testing_gradient << endl; \
        cout << endl; \
    }

#define TEST_GRADIENT_M(parameter_name) \
    { \
        for (size_t i = 0; i < transformer_input.parameter_name##_.row_count(); i++) \
        { \
            for (size_t j = 0; j < transformer_input.parameter_name##_.col_count(); j++) \
            { \
                TEST_GRADIENT_MI(parameter_name, i, j) \
            } \
        } \
    }

#define TEST_GRADIENT_V(parameter_name) \
    { \
        for (size_t i = 0; i < transformer_input.parameter_name##_.dim_count(); i++) \
        { \
            TEST_GRADIENT_VI(parameter_name, i) \
        } \
    }

    TEST_GRADIENT_M(all_token_embeddings)
    TEST_GRADIENT_M(all_position_embeddings)

    TEST_GRADIENT_V(gamma_X)
    TEST_GRADIENT_V(beta_X)

    TEST_GRADIENT_M(weight_Q)
    TEST_GRADIENT_M(weight_K)
    TEST_GRADIENT_M(weight_V)
    
    TEST_GRADIENT_V(gamma_A);
    TEST_GRADIENT_V(beta_A);
    
    TEST_GRADIENT_M(up_projector);
    TEST_GRADIENT_M(down_projector);
    TEST_GRADIENT_V(up_bias);
    TEST_GRADIENT_V(down_bias);

#undef TEST_GRADIENT_V
#undef TEST_GRADIENT_M
#undef TEST_GRADIENT_VI
#undef TEST_GRADIENT_MI

    return 0;
}
