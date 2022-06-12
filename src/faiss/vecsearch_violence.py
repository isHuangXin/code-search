import numpy as np
import time
import os, psutil



CosSim_dot = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class VecSearch:
    def __init__(self):
        self.dicts = {}

    def curr_items(self):
        return len(self.dicts)

    def add_doc(self, key, vector):
        self.dicts[key] = vector

    def search(self, query, top_k=10):
        ret = np.zeros((top_k, 2))

        for key, value in self.dicts.items():
            sim = CosSim_dot(query, value)
            if sim > ret[top_k-1][0]:
                b = np.array([[sim, key]]).astype('float32')
                ret = np.insert(ret, 0, values=b, axis=0)

                idex = np.lexsort([-1*ret[:, 0]])
                ret = ret[idex, :]
                ret = ret[:top_k, ]

        return ret[:, 0], ret[:, 1].astype('int')






if __name__ == "__main__":
    np.random.seed(1234)  # make reproducible
    print('大批量向量余弦相似度计算-[暴力版]'.center(40, '='), flush=True)
    total = 100000
    dim = 768
    print('随机生成%d个向量，维度：%d' % (total, dim), flush=True)
    xb = np.random.random((total, dim))
    xb[:, 0] += np.arange(total) / 1000.

    print('正在创建搜索器...')
    start = time.time()
    vs = VecSearch()
    for i in range(total):
        vs.add_doc(i, xb[i])
    end = time.time()
    total_time = end - start
    print('添加用时:%4f秒' % total_time)

    process = psutil.Process(os.getpid())
    print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')

    # for test
    print("单条查询测试".center(40, '-'))
    test_times = 100

    Q = np.random.random((test_times, dim))
    Q[:, 0] += np.arange(test_times) / 1000.

    q = Q[0]
    top_k = 10
    D, I = vs.search(q, top_k)
    print("搜索结果: ", D, I)

    def showdetail(x, q, D, I):
        print('显示查询结果，并验证余弦相似度...')
        for i, v in enumerate(I):
            r = (v, D[i])
            print('索引号:%5d, 距离:%f' % r)


    showdetail(xb, q, D, I)

    print('批量查询测试'.center(40, '-'))
    start = time.time()
    print('批量测试次数：%d 次，请稍候...' % test_times)
    for i in range(test_times):
        r = vs.search(Q[i])

    end = time.time()
    # print((end-start), (end-start)/test_times)
    total_time = end - start
    print('总用时:%d 秒, 平均用时:%4f 秒' % (total_time, total_time / test_times))

    # human evaluation
    while 1:
        print('-' * 40)
        txt = input("回车开始测试(Q退出): ").strip()
        if txt.upper() == "Q":
            break

        # 随机生成一个向量
        print("随机生成一个查询向量.. ")
        q = np.random.random(dim)
        print("query:%s..." % q[:10])

        # 查询
        start = time.time()
        r = vs.search(q)
        print("查询结果: ")
        print("索引号:%d, 相似度:%f" % r)
        end = time.time()
        total_time = end - start
        print('总用时:%d 秒, 平均用时:%4f 毫秒' % (total_time, total_time * 1000))
