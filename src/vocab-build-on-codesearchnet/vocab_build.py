import argparse
import json
import os
from collections import Counter


def json_read(data_path, language, type_dataset):
    code, desc, methname = [], [], []

    data_path = os.path.join(data_path, type_dataset)
    files = os.listdir(data_path)

    for i in range(len(files)):
        file_name = f'{language}_{type_dataset}_{i}.jsonl'
        print(f"loading... {language}_{type_dataset}_{i}.jsonl")
        file = open(os.path.join(data_path, file_name), 'r')
        for line in file.readlines():
            dic = json.loads(line)
            methname.append(dic['func_name'])
            code.append(dic['code_tokens'])
            desc.append(dic['docstring_tokens'])
    return methname, code, desc

def vocabulary_build(whitespace_split_tokens_list, dataset_len, vocab_len, type_dataset):
    vocabulary_list = []
    for idx, one_piece_code_tokens in enumerate(whitespace_split_tokens_list):
        print(f"processing... {type_dataset}/{idx}")
        for tokens in one_piece_code_tokens:
            tokens_dot_split = tokens.split(".")
            for tokens_wo_dot in tokens_dot_split:
                tokens_underline_split = tokens_wo_dot.split("_")
                finally_splited_tokens = list_camel_case_split(tokens_underline_split)
                vocabulary_list.extend(finally_splited_tokens)
        if dataset_len != -1:
            if idx >= dataset_len - 1:
                break
    vocabulary_frequency = dict(Counter(vocabulary_list))
    vocabulary = sorted(vocabulary_frequency.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    vocabulary_top = vocabulary[:vocab_len-4]
    vocabulary2idx = dict()
    vocabulary2idx['<pad>'] = 0
    vocabulary2idx['<s>'] = 1
    vocabulary2idx['</s>'] = 2
    vocabulary2idx['<unk>'] = 3

    idx2vocabulary = dict()
    idx2vocabulary[0] = '<pad>'
    idx2vocabulary[1] = '<s>'
    idx2vocabulary[2] = '</s>'
    idx2vocabulary[3] = '<unk>'

    for i in range(4, vocab_len):
        vocabulary2idx[vocabulary_top[i-4][0]] = i
        idx2vocabulary[i] = vocabulary_top[i-4][0]
    return vocabulary2idx, idx2vocabulary

def methname_vocabulary_build(whitespace_split_tokens_list, dataset_len, vocab_len, type_dataset):
    vocabulary_list = []
    for idx, tokens in enumerate(whitespace_split_tokens_list):
        print(f"processing ... {type_dataset}/{idx}")
        tokens_dot_split = tokens.split(".")
        for tokens_wo_dot in tokens_dot_split:
            tokens_underline_split = tokens_wo_dot.split("_")
            finally_splited_tokens = list_camel_case_split(tokens_underline_split)
            vocabulary_list.extend(finally_splited_tokens)
        if dataset_len != -1:
            if idx >= dataset_len-1:
                break
    vocabulary_frequency = dict(Counter(vocabulary_list))
    vocabulary = sorted(vocabulary_frequency.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    vocabulary_top = vocabulary[:vocab_len-4]
    vocabulary2idx = dict()
    vocabulary2idx['<pad>'] = 0
    vocabulary2idx['<s>'] = 1
    vocabulary2idx['</s>'] = 2
    vocabulary2idx['<unk>'] = 3

    idx2vocabulary = dict()
    idx2vocabulary[0] = '<pad>'
    idx2vocabulary[1] = '<s>'
    idx2vocabulary[2] = '</s>'
    idx2vocabulary[3] = '<unk>'

    for i in range(4, vocab_len):
        vocabulary2idx[vocabulary_top[i-4][0]] = i
        idx2vocabulary[i] = vocabulary_top[i-4][0]
    return vocabulary2idx, idx2vocabulary

def list_camel_case_split(words_list):
    result = []
    for str in words_list:
        if str == '':
            continue
        words = [[str[0]]]
        for c in str[1:]:
            if words[-1][-1].islower() and c.isupper():
                words.append(list(c))
            else:
                words[-1].append(c)
        result.extend([''.join(word) for word in words])
    return result


def json_file_read(path):
    with open(path, "r") as vocabulary_json_file:
        vocabulary_json = json.load(vocabulary_json_file)
    vocabulary_json_file.close()
    return vocabulary_json


def json_file_write(data, path):
    vocabulary_string = json.dumps(data)
    with open(path, "w") as vocabulary_json_file:
       vocabulary_json_file.write(vocabulary_string)
    vocabulary_json_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--data_path', type=str, default='/mnt/gold/huangxin/data/raw_codesearchnet/python', help='location of the data corpus')
    parser.add_argument('--language', type=str, default='python', help='dataset language')
    parser.add_argument('--train_dataset', type=str, default='train', help='dataset train')
    parser.add_argument('--valid_dataset', type=str, default='valid', help='dataset valid')
    parser.add_argument('--test_dataset', type=str, default='test', help='dataset test')
    parser.add_argument('--methname_dataset', type=str, default='methname', help='dataset train')
    parser.add_argument('--code_tokens_dataset', type=str, default='code_tokens', help='dataset valid')
    parser.add_argument('--desc_tokens_dataset', type=str, default='desc_tokens', help='dataset test')
    parser.add_argument('--dataset_len', type=int, default=-1, help='code length')
    parser.add_argument('--vocab_len', type=int, default=10000, help='code length')
    parser.add_argument('--dataset_dump_path', type=str, default="/mnt/gold/huangxin/data/raw_codesearchnet/my_vocab_processed_codesearchnet_in_pckl/python/my_vocab_on_csn_dot_final", help="path to dump pickle")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.dataset_dump_path, f"methname_tokens_vocabulary2idx_{args.vocab_len}.json")) and os.path.join(args.dataset_dump_path, f"methname_tokens_idx2vocabulary_{args.vocab_len}.json") \
            and os.path.exists(os.path.join(args.dataset_dump_path, f"code_tokens_vocabulary2idx_{args.vocab_len}.json")) and os.path.join(args.dataset_dump_path, f"code_tokens_idx2vocabulary_{args.vocab_len}.json") \
            and os.path.exists(os.path.join(args.dataset_dump_path, f"desc_tokens_vocabulary2idx_{args.vocab_len}.json")) and os.path.join(args.dataset_dump_path, f"desc_tokens_idx2vocabulary_{args.vocab_len}.json"):
        # loading... methname_tokens_vacabulary
        methname_tokens_vocabulary2idx = json_file_read(os.path.join(args.dataset_dump_path, f"methname_tokens_vocabulary2idx_{args.vocab_len}.json"))
        methname_tokens_idx2vocabulary = json_file_read(os.path.join(args.dataset_dump_path, f"methname_tokens_idx2vocabulary_{args.vocab_len}.json"))
        # loading... code_tokens_vacabulary
        code_tokens_vocabulary2idx = json_file_read(os.path.join(args.dataset_dump_path, f"code_tokens_vocabulary2idx_{args.vocab_len}.json"))
        code_tokens_idx2vocabulary = json_file_read(os.path.join(args.dataset_dump_path, f"code_tokens_idx2vocabulary_{args.vocab_len}.json"))
        # loading... desc_tokens_vacabulary
        desc_tokens_vocabulary2idx = json_file_read(os.path.join(args.dataset_dump_path, f"desc_tokens_vocabulary2idx_{args.vocab_len}.json"))
        desc_tokens_idx2vocabulary = json_file_read(os.path.join(args.dataset_dump_path, f"desc_tokens_idx2vocabulary_{args.vocab_len}.json"))
    else:
        code_tokens, methname_tokens, desc_tokens = [], [], []
        train_methname, train_code_tokens, train_desc_tokens = json_read(args.data_path, args.language, args.train_dataset)
        valid_methname, valid_code_tokens, valid_desc_tokens = json_read(args.data_path, args.language, args.valid_dataset)
        test_methname, test_code_tokens, test_desc_tokens = json_read(args.data_path, args.language, args.test_dataset)

        methname_tokens.extend(train_methname)
        methname_tokens.extend(valid_methname)
        methname_tokens.extend(test_methname)
        methname_tokens_vocabulary2idx, methname_tokens_idx2vocabulary = methname_vocabulary_build(methname_tokens, args.dataset_len, args.vocab_len, args.methname_dataset)
        json_file_write(methname_tokens_vocabulary2idx, os.path.join(args.dataset_dump_path, f"methname_tokens_vocabulary2idx_{args.vocab_len}.json"))
        json_file_write(methname_tokens_idx2vocabulary, os.path.join(args.dataset_dump_path, f"methname_tokens_idx2vocabulary_{args.vocab_len}.json"))

        code_tokens.extend(train_code_tokens)
        code_tokens.extend(valid_code_tokens)
        code_tokens.extend(test_code_tokens)
        code_tokens_vocabulary2idx, code_tokens_idx2vocabulary = vocabulary_build(code_tokens, args.dataset_len, args.vocab_len, args.code_tokens_dataset)
        json_file_write(code_tokens_vocabulary2idx, os.path.join(args.dataset_dump_path, f"code_tokens_vocabulary2idx_{args.vocab_len}.json"))
        json_file_write(code_tokens_idx2vocabulary, os.path.join(args.dataset_dump_path, f"code_tokens_idx2vocabulary_{args.vocab_len}.json"))

        desc_tokens.extend(train_desc_tokens)
        desc_tokens.extend(valid_desc_tokens)
        desc_tokens.extend(test_desc_tokens)
        desc_tokens_vocabulary2idx, desc_tokens_idx2vocabulary = vocabulary_build(desc_tokens, args.dataset_len, args.vocab_len, args.desc_tokens_dataset)
        json_file_write(desc_tokens_vocabulary2idx, os.path.join(args.dataset_dump_path, f"desc_tokens_vocabulary2idx_{args.vocab_len}.json"))
        json_file_write(desc_tokens_idx2vocabulary, os.path.join(args.dataset_dump_path, f"desc_tokens_idx2vocabulary_{args.vocab_len}.json"))
    print("done!")
