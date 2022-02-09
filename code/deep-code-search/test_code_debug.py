

f = open("/mnt/wanyao/huangxin/data/deep_code_search/example/test.desc.txt","r")
lines = f.readlines()   # 读取全部内容 ，并以列表方式返回
for line in lines:
    print(line)