import argparse
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


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
        # break
    return methname, code, desc

def underline_camel_split_embedding(methname_list, code_tokens_list, desc_tokens_list, methname_len, code_len, desc_len, type_dataset):

    methname_vocabulary2idx = json_file_read(os.path.join(args.vocab_dump_path, f"methname_tokens_vocabulary2idx_{args.vocab_len}.json"))
    code_tokens_vocabulary2idx = json_file_read(os.path.join(args.vocab_dump_path, f"code_tokens_vocabulary2idx_{args.vocab_len}.json"))
    desc_tokens_vocabulary2idx = json_file_read(os.path.join(args.vocab_dump_path, f"desc_tokens_vocabulary2idx_{args.vocab_len}.json"))

    methname_repr_vec = []
    code_repr_vec = []
    desc_repr_vec = []

    source_methname_tokens_len = []
    source_code_tokens_len = []
    source_desc_tokens_len = []

    for i in range(len(methname_list)):
        print(f"processing... {type_dataset}/{i} methname, code_tokens, desc_tokens")

        one_methname_split = []
        one_methname_repr_vec = []
        methname_tokens_dot_split = methname_list[i].split(".")
        for methname_tokens_wo_dot in methname_tokens_dot_split:
            methname_tokens_underline_split = methname_tokens_wo_dot.split("_")
            one_methname_split.extend(list_camel_case_split(methname_tokens_underline_split))
        for methname_tokens_split in one_methname_split:
            if methname_tokens_split in methname_vocabulary2idx:
                one_methname_repr_vec.append(methname_vocabulary2idx[methname_tokens_split])
            else:
                one_methname_repr_vec.append(methname_vocabulary2idx['<unk>'])
        source_methname_tokens_len.append(len(one_methname_split))
        methname_repr_vec.append(np.array(one_methname_repr_vec[:methname_len]))


        one_code_tokens_split = []
        one_code_repr_vec = []
        for code_tokens in code_tokens_list[i]:
            code_tokens_dot_split = code_tokens.split(".")
            for code_tokens_wo_dot in code_tokens_dot_split:
                code_tokens_underline_split = code_tokens_wo_dot.split("_")
                one_code_tokens_split.extend(list_camel_case_split(code_tokens_underline_split))
        for code_tokens_split in one_code_tokens_split:
            if code_tokens_split in code_tokens_vocabulary2idx:
                one_code_repr_vec.append(code_tokens_vocabulary2idx[code_tokens_split])
            else:
                one_code_repr_vec.append(code_tokens_vocabulary2idx['<unk>'])
        source_code_tokens_len.append(len(one_code_tokens_split))
        code_repr_vec.append(np.array(one_code_repr_vec[:code_len]))


        one_desc_tokens_split = []
        one_desc_repr_vec = []
        for desc_tokens in desc_tokens_list[i]:
            desc_tokens_dot_split = desc_tokens.split(".")
            for desc_tokens_wo_dot in desc_tokens_dot_split:
                desc_tokens_underline_split = desc_tokens_wo_dot.split("_")
                one_desc_tokens_split.extend(list_camel_case_split(desc_tokens_underline_split))
        for desc_tokens_split in one_desc_tokens_split:
            if desc_tokens_split in desc_tokens_vocabulary2idx:
                one_desc_repr_vec.append(desc_tokens_vocabulary2idx[desc_tokens_split])
            else:
                one_desc_repr_vec.append(desc_tokens_vocabulary2idx['<unk>'])
        source_desc_tokens_len.append(len(one_desc_tokens_split))
        desc_repr_vec.append(np.array(one_desc_repr_vec[:desc_len]))

        if args.dataset_len != -1:
            if i >= args.dataset_len - 1:
                break
    return methname_repr_vec, code_repr_vec, desc_repr_vec, source_methname_tokens_len, source_code_tokens_len, source_desc_tokens_len


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


def statistics_len(code_len_list, desc_len_list, methname_len_list, dataset_dump_path, type_dataset):
    code_num = len(code_len_list)
    # index_list = [i for i in range(0, code_num)]
    # plt.bar(index_list, code_len_list)
    # plt.title(f'{type_dataset} code_len_statistics')
    # plt.xlabel("index")
    # plt.ylabel("length")
    # plt.savefig(os.path.join(dataset_dump_path, "figure", f"{type_dataset}_code_len_statistics.pdf"))
    print(f"{type_dataset} code tokens length statistics")
    code_below_150 = sum([i < 100 for i in code_len_list])
    code_below_200 = sum([i < 150 for i in code_len_list])
    code_below_250 = sum([i < 200 for i in code_len_list])
    code_below_300 = sum([i < 250 for i in code_len_list])
    print(f"code_below_150: {code_below_150/(1.0 * code_num) * 100}%")
    print(f"code_below_200: {code_below_200/(1.0 * code_num) * 100}%")
    print(f"code_below_250: {code_below_250/(1.0 * code_num) * 100}%")
    print(f"code_below_300: {code_below_300/(1.0 * code_num) * 100}%")
    print("\n")
    # plt.show()

    # plt.bar(index_list, desc_len_list)
    # plt.title(f'{type_dataset} desc_len_statistics')
    # plt.xlabel("index")
    # plt.ylabel("length")
    # plt.savefig(os.path.join(dataset_dump_path, "figure", f"{type_dataset}_desc_len_statistics.pdf"))
    print(f"{type_dataset} desc tokens length statistics")
    desc_below_25 = sum([i < 25 for i in desc_len_list])
    desc_below_50 = sum([i < 50 for i in desc_len_list])
    desc_below_75 = sum([i < 75 for i in desc_len_list])
    desc_below_100 = sum([i < 100 for i in desc_len_list])
    print(f"desc_below_25: {desc_below_25 / (1.0 * code_num) * 100}%")
    print(f"desc_below_50: {desc_below_50 / (1.0 * code_num) * 100}%")
    print(f"desc_below_75: {desc_below_75 / (1.0 * code_num) * 100}%")
    print(f"desc_below_100: {desc_below_100 / (1.0 * code_num) * 100}%")
    print("\n")
    # plt.show()

    # plt.bar(index_list, methname_len_list)
    # plt.title(f'{type_dataset} methname_len_statistics')
    # plt.xlabel("index")
    # plt.ylabel("length")
    # plt.savefig(os.path.join(dataset_dump_path, "figure", f"{type_dataset}_methname_len_statistics.pdf"))
    print(f"{type_dataset} methname tokens length statistics")
    methname_below_6 = sum([i < 6 for i in methname_len_list])
    methname_below_8 = sum([i < 8 for i in methname_len_list])
    methname_below_10 = sum([i < 10 for i in methname_len_list])
    methname_below_12 = sum([i < 12 for i in methname_len_list])
    print(f"methname_below_6: {methname_below_6 / (1.0 * code_num) * 100}%")
    print(f"methname_below_8: {methname_below_8 / (1.0 * code_num) * 100}%")
    print(f"methname_below_10: {methname_below_10 / (1.0 * code_num) * 100}%")
    print(f"methname_below_12: {methname_below_12 / (1.0 * code_num) * 100}%")
    # plt.show()
    print("done!\n")


def pickle_file_read(path):
    with open(os.path.join(path), "rb") as repr_file:
        repr_file_pickl = pickle.load(repr_file)
    repr_file.close()
    return repr_file_pickl


def pickle_file_write(data, path):
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()


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
    parser.add_argument('--methname_len', type=int, default='8', help='methname length')
    parser.add_argument('--code_len', type=int, default='250', help='code length')
    parser.add_argument('--desc_len', type=int, default='50', help='query length')
    parser.add_argument('--vocab_len', type=int, default=10000, help='code length')
    parser.add_argument('--dataset_len', type=int, default=-1, help='code length')
    parser.add_argument('--vocab_dump_path', type=str, default="/mnt/gold/huangxin/data/raw_codesearchnet/my_vocab_processed_codesearchnet_in_pckl/python/my_vocab_on_csn_dot_final",help="path to dump pickle")
    parser.add_argument('--dataset_dump_path', type=str, default="/mnt/gold/huangxin/data/raw_codesearchnet/my_vocab_processed_codesearchnet_in_pckl/python/codetokens_len_250_final", help="path to dump pickle")
    args = parser.parse_args()


    if os.path.exists(os.path.join(args.dataset_dump_path, f"train_methname_{args.dataset_len}_{args.methname_len}.pkl")) and os.path.join(args.dataset_dump_path, f"train_codetokens_{args.dataset_len}_{args.code_len}.pkl") \
            and os.path.exists(os.path.join(args.dataset_dump_path, f"train_desctokens_{args.dataset_len}_{args.desc_len}.pkl")):
        train_methname_repr_pickl = pickle_file_read(os.path.join(args.dataset_dump_path, f"train_methname_{args.dataset_len}_{args.methname_len}.pkl"))
        train_code_repr_pickl = pickle_file_read(os.path.join(args.dataset_dump_path, f"train_codetokens_{args.dataset_len}_{args.code_len}.pkl"))
        train_desc_repr_pickl = pickle_file_read(os.path.join(args.dataset_dump_path, f"train_desctokens_{args.dataset_len}_{args.desc_len}.pkl"))
    else:
        train_methname, train_code, train_desc = json_read(args.data_path, args.language, args.train_dataset)
        train_methname_repr, train_code_repr, train_desc_repr,\
            train_methname_tokens_len, train_code_tokens_len, train_desc_tokens_len \
            = underline_camel_split_embedding(train_methname, train_code, train_desc, args.methname_len, args.code_len, args.desc_len, args.train_dataset)

        # statistics_len(train_code_tokens_len, train_desc_tokens_len, train_methname_tokens_len, args.dataset_dump_path, args.train_dataset)
        # print(f"{args.train_dataset} statistics done!")
        pickle_file_write(train_methname_repr, os.path.join(args.dataset_dump_path, f"train_methname_{args.dataset_len}_{args.methname_len}.pkl"))
        pickle_file_write(train_code_repr, os.path.join(args.dataset_dump_path, f"train_codetokens_{args.dataset_len}_{args.code_len}.pkl"))
        pickle_file_write(train_desc_repr, os.path.join(args.dataset_dump_path, f"train_desctokens_{args.dataset_len}_{args.desc_len}.pkl"))


    if os.path.exists(os.path.join(args.dataset_dump_path, f"valid_methname_{args.dataset_len}_{args.methname_len}.pkl")) and os.path.join(args.dataset_dump_path, f"valid_codetokens_{args.dataset_len}_{args.code_len}.pkl") \
            and os.path.exists(os.path.join(args.dataset_dump_path, f"valid_desctokens_{args.dataset_len}_{args.desc_len}.pkl")):
        valid_methname_repr_pickl = pickle_file_read(os.path.join(args.dataset_dump_path, f"valid_methname_{args.dataset_len}_{args.methname_len}.pkl"))
        valid_code_repr_pickl = pickle_file_read(os.path.join(args.dataset_dump_path, f"valid_codetokens_{args.dataset_len}_{args.code_len}.pkl"))
        valid_desc_repr_pickl = pickle_file_read(os.path.join(args.dataset_dump_path, f"valid_desctokens_{args.dataset_len}_{args.desc_len}.pkl"))
    else:
        valid_methname, valid_code, valid_desc = json_read(args.data_path, args.language, args.valid_dataset)
        valid_methname_repr, valid_code_repr, valid_desc_repr,\
            valid_methname_tokens_len, valid_code_tokens_len, valid_desc_tokens_len \
            = underline_camel_split_embedding(valid_methname, valid_code, valid_desc, args.methname_len, args.code_len, args.desc_len, args.valid_dataset)

        # statistics_len(valid_code_tokens_len, valid_desc_tokens_len, valid_methname_tokens_len, args.dataset_dump_path, args.valid_dataset)
        # print(f"{args.train_dataset} statistics done!")
        pickle_file_write(valid_methname_repr, os.path.join(args.dataset_dump_path, f"valid_methname_{args.dataset_len}_{args.methname_len}.pkl"))
        pickle_file_write(valid_code_repr, os.path.join(args.dataset_dump_path, f"valid_codetokens_{args.dataset_len}_{args.code_len}.pkl"))
        pickle_file_write(valid_desc_repr, os.path.join(args.dataset_dump_path, f"valid_desctokens_{args.dataset_len}_{args.desc_len}.pkl"))


    if os.path.exists(os.path.join(args.dataset_dump_path, f"test_methname_{args.dataset_len}_{args.methname_len}.pkl")) and os.path.join(args.dataset_dump_path, f"test_codetokens_{args.dataset_len}_{args.code_len}.pkl") \
            and os.path.exists(os.path.join(args.dataset_dump_path, f"test_desctokens_{args.dataset_len}_{args.desc_len}.pkl")):
        test_methname_repr_pickl = pickle_file_read(os.path.join(args.dataset_dump_path, f"test_methname_{args.dataset_len}_{args.methname_len}.pkl"))
        test_code_repr_pickl = pickle_file_read(os.path.join(args.dataset_dump_path, f"test_codetokens_{args.dataset_len}_{args.code_len}.pkl"))
        test_desc_repr_pickl = pickle_file_read(os.path.join(args.dataset_dump_path, f"test_desctokens_{args.dataset_len}_{args.desc_len}.pkl"))
    else:
        test_methname, test_code, test_desc = json_read(args.data_path, args.language, args.test_dataset)
        test_methname_repr, test_code_repr, test_desc_repr,\
            test_methname_tokens_len, test_code_tokens_len, test_desc_tokens_len \
            = underline_camel_split_embedding(test_methname, test_code, test_desc, args.methname_len, args.code_len, args.desc_len, args.test_dataset)

        # statistics_len(test_code_tokens_len, test_desc_tokens_len, test_methname_tokens_len, args.dataset_dump_path, args.test_dataset)
        # print(f"{args.test_dataset} statistics done!")
        pickle_file_write(test_methname_repr, os.path.join(args.dataset_dump_path, f"test_methname_{args.dataset_len}_{args.methname_len}.pkl"))
        pickle_file_write(test_code_repr, os.path.join(args.dataset_dump_path, f"test_codetokens_{args.dataset_len}_{args.code_len}.pkl"))
        pickle_file_write(test_desc_repr, os.path.join(args.dataset_dump_path, f"test_desctokens_{args.dataset_len}_{args.desc_len}.pkl"))

    print("data embedding done!\n")
    print(f"{args.train_dataset} statistics begin!")
    statistics_len(train_code_tokens_len, train_desc_tokens_len, train_methname_tokens_len, args.dataset_dump_path, args.train_dataset)
    print(f"{args.valid_dataset} statistics begin!")
    statistics_len(valid_code_tokens_len, valid_desc_tokens_len, valid_methname_tokens_len, args.dataset_dump_path, args.valid_dataset)
    print(f"{args.test_dataset} statistics begin!")
    statistics_len(test_code_tokens_len, test_desc_tokens_len, test_methname_tokens_len, args.dataset_dump_path, args.test_dataset)
    print("done!")
