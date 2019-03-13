import csv, math
import numpy as np
from collections import defaultdict
from scipy.stats import norm


def readData(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        attributes = next(reader)[1:9]
        dataset = []
        for line in reader:
            dataset.append(line[1:10])
    # 返回数据集列表及属性列表
    print('dataset = {}'.format(dataset))
    print('attributes = {}'.format(attributes))
    return dataset, attributes


def classify(dataset):
    T = []
    F = []
    for vec in dataset:
        if vec[-1] == '是':
            T.append(vec)
        elif vec[-1] == '否':
            F.append(vec)
    return T, F


def p_dic(dataset, cset, attris):
    p_dic = defaultdict(int)
    for index in range(len(attris)):
        p = defaultdict(float)
        attri_values = [vec[index] for vec in dataset]
        if all(c in '0123456789.-' for c in attri_values[0]):  # 连续值属性
            attri_values = [float(i) for i in attri_values]
            u = np.mean(attri_values)
            v = np.std(attri_values, ddof=1)  # 样本标准差
            p['mean'] = u
            p['std'] = v
        else:
            attri_values = set(attri_values)
            for a in attri_values:
                c = 0
                for vec in cset:
                    if vec[index] == a:
                        c += 1
                p[a] = float((c+1)/(len(cset)+len(attri_values)))
        p_dic[attris[index]] = p
    return p_dic


def predic(p_dic, testdata):
    p_xi_c = 1
    index = 0
    for key in p_dic.keys():
        # print(p_dic[key])
        if 'mean'and 'std'in p_dic[key]:  # 连续值
            # print('continuos')
            p_xi = (np.exp(-((float(testdata[index])-float(p_dic[key]['mean']))**2)
                          /2*(float(p_dic[key]['std'])**2)))/(((2*math.pi)**0.5)*(float(p_dic[key]['std'])))
            # print(p_xi)
            # p_xi = norm(float(p_dic[key]['mean']), float(p_dic[key]['std'])).cdf(float(testdata[index]))
            # print(p_xi)
        else:
            p_xi = float(p_dic[key].setdefault(testdata[index]))
        p_xi_c *= p_xi
        index += 1
    return p_xi_c


if __name__ == '__main__':
    file_path = 'data\watermelon3_0_Ch.csv'
    data, attributes = readData(file_path)
    T, F = classify(data)
    p_T = p_dic(data, T, attributes)
    p_F = p_dic(data, F, attributes)

    test = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.697', '0.46']
    p_t = predic(p_T, test)
    p_f = predic(p_F, test)
    print('p_t = {}'.format(p_t))
    print('p_f = {}'.format(p_f))

