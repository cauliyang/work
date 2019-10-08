# coding=utf-8

''' 
      K_mean for bioinfomatic's project 

-------------------------------------------------
Requirement : numpy 

Usage : python k_mean.py k data max_it (cetroids)

Result_file : kmeans.out

Contact : <liyangyang> <liyangyangz5@163.com>

-------------------------------------------------

'''

# 导入模块
import sys
import time
import numpy as np
from collections import Counter
from operator import itemgetter


def eucl_Distance(init_cetroids, piece_data):
    ''' 
    计算每一个数据与质心的欧式距离
    '''
    distance = np.sqrt(np.sum((init_cetroids - piece_data) ** 2,
                              axis=1))
    # 返回欧式距离
    return distance


def summary(kw, tim, kmeanout='kmeans.out'):
    '''
    创建总结函数，统计递归次数，运行时间等信息。
    '''
    # 创建统计各Cluster数据量函数
    def print_cluster(kmean=kmeanout):
        # 统计数据
        counter = Counter(np.loadtxt(kmean, dtype=int)[:, 1])
        # 生成报告
        for clu, num in counter.most_common():
            print(f'    Cluster_{clu} : {num}')
    # 创建统计表头
    print('{:-^40}\n'.format('Summary'))
    # 生成各Cluster数据量统计报告
    print_cluster()
    # 描述信息
    print(f'''
    Max_iter_number : {kw['max_it']} 
    Cluster_number  :{kw['k']} 
    Time  : {tim:.2f}s 
    Date  : {time.asctime()} 
    ''')
    # 创建统计表尾
    print('{:-<40}\n'.format('-'))


def get_Cetroid(file, k, cetroid_file=None):
    ''' 
    此函数用于获取原始数据文件信息：原始数据，质心，数据量，特征维度
    '''
    # 读取文件
    data = np.loadtxt(file)
    # 获取文件信息：数据量，特征维度
    gene_num, ndim = data.shape
    # 判断用户是否提供质心，如未提供则随机产生
    if not(cetroid_file):
        # 初始化质心
        init_cetroids = np.zeros((k, ndim))
        # 随机产生质心
        for i in range(k):
            index = int(np.random.uniform(0, gene_num))
            init_cetroids[i, :] = data[index, :]
    else:
        # 若用户提供则读取质心
        init_cetroids = np.loadtxt(cetroid_file)
    # 返回所需要的信息：原始数据，质心，数据量，特征纬度
    return (data, init_cetroids, gene_num, ndim)


def get_argv():
    '''
    获取用户输入的参数，并返回字典型参数
    '''
    # 获取用户输入
    argv_list = sys.argv
    # 初始化参数
    argv_name = ('data',
                 'init_cetroids',
                 'gene_num',
                 'ndim',
                 'max_it',
                 'k')
    # 判断用户是否提供质心文件，根据参数个数(不是很严谨，没有捕捉错误)
    if len(argv_list) == 4:
        # 用户没有提供质心文件
        _, k, file, max_it = argv_list
        # 获取数据信息
        argv_tuple = get_Cetroid(file, int(k)) + (int(max_it), int(k))
    elif len(argv_list) == 5:
        # 用户提供质心文件
        _, k, file, max_it, cetroid_file = argv_list
        # 获取数据信息
        argv_tuple = get_Cetroid(file, int(k),
                                 cetroid_file=cetroid_file) + (int(max_it), int(k))
    elif len(argv_list) < 4:
        # 提示用户输入错误
        print('''
            -------------------------------------------------
            Requirement : numpy 

            Usage : python k_mean.py k data max_it (cetroids)

            Result_file : kmeans.out

            Contact : <liyangyang> <liyangyangz5@163.com>

            -------------------------------------------------

            ''')
        sys.exit(0)
    # 返回字典型参数
    return dict(zip(argv_name, argv_tuple))


def assert_Result(data, init_cetroids, result, k):
    '''
    重新计算质心，并判断迭代的结果是否稳定 
    '''
    # 初始化新质心
    init_cetroid_new = np.zeros_like(init_cetroids)
    # 计算新的质心
    for i in range(k):
        index = result[result[:, 1] == i][:, 0]
        init_cetroid_new[i] = np.mean(data[index], axis=0)
    init_cetroid_new = np.round(init_cetroid_new, decimals=2)
    # 判断质心是否变化，并返回新的质心。
    return init_cetroid_new == init_cetroids, init_cetroid_new


def iter_Cetroid(**argv):
    '''
    迭代聚类的结果
    '''
    # 获取所需的数据信息
    data, init_cetroids, gene_num, ndim, max_it, k = argv.values()
    # 初始化结果
    Result = np.zeros((gene_num, 2), dtype=int)
    # 根据欧式距离分群
    for i in range(gene_num):
        # 获取欧式距离
        distance = eucl_Distance(init_cetroids, data[i, :])
        # 获取距离最短cluster的label
        cluster = distance.argmin()
        # 进行分群
        Result[i, :] = np.array([i, cluster])
    # 验证迭代的结果是否稳定，并返回新的质心
    Handle, argv['init_cetroids'] = assert_Result(data,
                                                  init_cetroids, Result, k)
    # 返回结果，验证结果，字典型参数，最大迭代数
    return Result, Handle.all(), argv, max_it


def run(arg_dict, it_num=0):
    '''
    k_mean 主体函数，判断迭代次数是否达到最大迭代次数，以及是否提前分群成功
    '''
    # 进行一次迭代并验证结果是否稳定，并计算新的质心以字典型返回。
    Result, handle, arg_dict, max_it = iter_Cetroid(**arg_dict)
    # 判断是否达到结束迭代条件
    if not(handle) and (it_num < max_it):
        # 没有达到终止迭代的条件，继续迭代。
        it_num += 1
        # 提示迭代次数
        print(f'...ing Iter Number :{it_num}')
        # 递归迭代
        run(arg_dict, it_num=it_num)
    # 达到迭代结束条件
    else:
        # 将数字起始改为1
        Result = Result + 1
        count_1 = Counter(Result[:, 1])
        # 保存结果文件
        np.savetxt('kmeans.out', Result, fmt='%d')
   # 返回最大迭代次数


def main():
    '''
    程序主函数，整合工作流程，并生成报告。
    '''
    # 获取开始时间
    TIC = time.time()
    # 获取字典型参数
    ARGV = get_argv()
    # 运行k-mean主体，并返回最大迭代数
    run(ARGV)
    # 获取结束时间
    TOC = time.time()
    # 生成报告
    summary(ARGV, TOC - TIC)


if __name__ == '__main__':
    main()
