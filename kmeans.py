import xlrd
import random
import numpy
from collections import defaultdict

# 计算欧式距离
def distance(xi, xj):
    s = 0
    for i in range(len(xi)):
        s += (xi[i]-xj[i])**2
    return s**0.5


def meanVec(dataset):
    v = []
    for i in range(len(dataset[0])):  # 0,1
        sum = 0
        for j in range(len(dataset)):   # 0 - m
            sum += dataset[j][i]
        v.append(sum/len(dataset))
    return v

def clustering(dataset, U):
    C = defaultdict(list)
    for i in dataset :
        index = numpy.argmin( map(lambda j: distance(i, j), U) )
        C[index].append(i)
    return C


def kmeans(dataset, k=3):
    # U = [dataset[5], dataset[11], dataset[23]]
    # 获取初始均值向量
    U = random.sample(dataset, k)
    count = 0
    for count in range(5):
        C = clustering(dataset, U)
        print('第{}次的 U = {}\n'.format(count, U))
        print('第{}次的 C = {}\n'.format(count, C))
        new_U = list(map(lambda i:meanVec(C[i]), C))
        if new_U == U :
            return C
        else :
            U = new_U
    return {}


if __name__ == '__main__':
    f = xlrd.open_workbook('data/watermelon4.0.xlsx').sheets()[0]
    dataset = []
    for row in range(f.nrows):
        dataset.append(f.row_values(row))
    attributes = ['密度', '含糖率']
    print('dataset = {}'.format(dataset))
    print('attributes = {}'.format(attributes))
    C = kmeans(dataset)
    print(C)

    # d = [[0.634, 0.264], [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267], [0.639, 0.161], [0.657, 0.198], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257], [0.483, 0.312]]
    # v = meanVec(d)
    # print(v)



