import argparse
import numpy as np
import time
import os, psutil

import torch
import torch.nn.functional as F


def get_cos_similar_multi(v1: list, v2: list, top_k):
    cos_sim = F.cosine_similarity(v1.unsqueeze(0), v2)
    cos_sim = cos_sim.numpy()

    index = list(range(len(cos_sim)))
    dict_cos_sim = dict(zip(index, cos_sim))
    sorted_cos_sim = sorted(dict_cos_sim.items(), key=lambda x: x[1], reverse=True)
    result = sorted_cos_sim[:top_k]
    return result

def get_cos_similar_matrix(v1, v2, top_k):
    cos_sim_matrix = F.cosine_similarity(v1.unsqueeze(1), v2.unsqueeze(0), dim=-1)
    cos_sim_matrix = cos_sim_matrix.numpy()
    result = []
    for i in range(len(cos_sim_matrix)):
        index = list(range(cos_sim_matrix.shape[1]))
        dict_cos_sim = dict(zip(index, cos_sim_matrix[i]))
        sorted_cos_sim = sorted(dict_cos_sim.items(), key=lambda x: x[1], reverse=True)
        result.append(sorted_cos_sim[:top_k])
    return result

def showdetail(query_result):
    print('显示查询结果，并验证余弦相似度...')
    for i in range(len(query_result)):
        r = (query_result[i][0], query_result[i][1])
        print('索引号:%5d, 余玄相似度:%f' % r)

def numpy_test(args):
    np.random.seed(args.seed)
    print('大批量向量余弦相似度计算-[暴力版]'.center(40, '='), flush=True)
    print('随机生成%d个向量，维度：%d' % (args.total, args.dim), flush=True)
    # numpy生成[a, b)区间的随机数, (b - a) * random_sample() + a
    x_b = 2 * np.random.random((args.total, args.dim)) - 1
    x_b = x_b.astype(np.float32)
    x_b = torch.from_numpy(x_b)

    # for test
    x_q = 2 * np.random.random((args.test_times, args.dim)) - 1
    x_q = x_q.astype(np.float32)
    x_q = torch.from_numpy(x_q)

    process = psutil.Process(os.getpid())
    print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')

    print("单条查询测试".center(40, '-'))
    q = x_q[0]
    print('正在进行单条数据搜索...')
    start = time.time()
    result = get_cos_similar_multi(q, x_b, args.top_k)
    end = time.time()
    total_time = end - start
    print('搜索用时:%4f秒' % total_time)
    showdetail(result)

    print('批量查询测试'.center(40, '-'))
    start = time.time()
    print('批量测试次数：%d 次，请稍候...' % args.test_times)
    get_cos_similar_matrix(x_q, x_b, args.top_k)
    end = time.time()
    total_time = end - start
    print('总用时:%d 秒, 平均用时:%4f 秒' % (total_time, total_time / args.test_times))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='faiss速度测试工具')
    parser.add_argument('--total', default=400000, type=int, help='总数据量,默认10万')  # required=True,
    parser.add_argument('--dim', default=768, type=int, help='向量维度,默认768')
    parser.add_argument('--test_times', default=100, type=int, help='测试次数，默认1万')
    parser.add_argument('--top_k', default=10, type=int, help='每次返回条数,默认5')
    parser.add_argument('--seed', default=1234, type=int, help='make reproducible')
    args = parser.parse_args()
    print(args)
    numpy_test(args)