#include <iostream>
#include <cassert>
#include <unistd.h>

#include "common/util.h"
#include "tokenizer/bpe_trainer.h"

using namespace std;
using namespace util;
using namespace tokenizer;

int main(int argc, char *argv[])
{
    int    opt = 0;
    size_t max_scanning_times = 10;
    size_t target_token_count = 3000;
    string training_corpus_path_filename = "";
    string trained_tokens_path_filename = "";

    char *dummy = nullptr;
    while ((opt = getopt(argc, argv, "t:c:i:o:")) != -1)
    {
        switch (opt)
        {
            case 't':
                max_scanning_times = strtoul(optarg, &dummy, 10);
                break;
            case 'c':
                target_token_count = strtoul(optarg, &dummy, 10);
                break;
            case 'i':
                training_corpus_path_filename = optarg;
                break;
            case 'o':
                trained_tokens_path_filename = optarg;
                break;
            case '?':
            default:
                cerr << "Invalid Parameters! Should be \"t:c:i:o:\"" << endl;
                return -1;
        }
    }
    
    assert(max_scanning_times > 0);
    assert(target_token_count > 0);
    assert(training_corpus_path_filename != "");
    assert(trained_tokens_path_filename != "");

    string training_corpus;
    int retcode = read_string_from_file(training_corpus_path_filename, training_corpus);
    if (retcode < 0)
    {
        cerr << "read_string_from_file [" << training_corpus_path_filename << "] Failed [" << retcode << "]." << endl;
        return -2;
    }

    BPE_Trainer trainer(max_scanning_times, target_token_count);
    vector<string> trained_tokens;
    retcode = trainer.train(training_corpus, trained_tokens);
    if (retcode < 0)
    {
        cerr << "trainer.train Failed [" << retcode << "]." << endl;
        return -3;
    }

    string trained_tokens_serialized;
    for (auto trained_token : trained_tokens)
    {
        trained_tokens_serialized += trained_token;
        trained_tokens_serialized += "\n";
    }

    retcode = dump_string_to_file(trained_tokens_serialized, trained_tokens_path_filename);
    if (retcode < 0)
    {
        cerr << "dump_string_to_file [" << training_corpus_path_filename << "] Failed [" << retcode << "]." << endl;
        return -4;
    }
    
    return 0;
}
