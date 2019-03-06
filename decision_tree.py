import numpy as np
import math, csv, operator
import pandas as pd


# 读取数据
def readData(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        atrributes = next(reader)[1:9]
        dataset = []
        for line in reader:
            dataset.append(line[1:10])
    # 返回数据集列表及属性列表
    return dataset, atrributes

# 计算当前输入数据集的信息熵Ent
def Ent(dataset):
    labels = {}
    for vec in dataset:
        label = vec[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    ent = 0
    for l in labels.keys():
        p = float(labels[l])/float(len(dataset))
        ent -= (p*np.log2(p))
    return ent


# 基于某个属性 划分数据集
def splitDataset(dataset, index, value, is_continuous, part=0):
    sub_dataset = []
    if is_continuous:  # index=6/7
        for vec in dataset:
            # 根据参数part，将连续值属性划分为两个部分分别输出
            if float(vec[index]) <= value and part == 0:
                # sub_vec = vec[:index].extend(vec[index+1:])
                sub_dataset.append(vec)
            elif float(vec[index]) > value and part == 1:
                # sub_vec = vec[:index].extend(vec[index+1:])
                sub_dataset.append(vec)
    else:
        for vec in dataset:
            if str(vec[index]) == str(value):
                # sub_vec = vec[:index].extend(vec[index + 1:])
                sub_dataset.append(vec)
    return sub_dataset


def attriToSplit(dataset):
    ent_D = Ent(dataset)   # 根节点的信息熵
    max_gain = 0    # 信息增益
    attri_to_split = -1
    split_value = None   # 保存最优划分点，处理连续值属性
    is_continuous = False
    for index in range(len(dataset[0])-1):  # 最后一个值为y标签，而非属性值
        # 每一轮循环，处理一个属性
        attri_values = [e[index] for e in dataset]   # 获取某属性的所有值的列表
        new_ent = 0
        # 处理连续值属性
        if all(c in '0123456789.-' for c in attri_values[0]):
            is_continuous = True
            attri_values.sort()
            value_list = [float(v) for v in attri_values]
            mid_point_values = []   # 计算所有划分候选点
            for i in range(len(value_list)-1):
                mid_point_values.append((value_list[i]+value_list[i+1])/2)
            # 遍历所有划分候选点，找出信息增益最大的划分值
            for value in mid_point_values:
                for part in range(2):
                    sub_dataset = splitDataset(dataset, index, value, is_continuous, part)
                    p = len(sub_dataset)/len(dataset)
                    new_ent += p*Ent(sub_dataset)
                gain = ent_D - new_ent
                if gain > max_gain:
                    max_gain = gain
                    attri_to_split = index
                    split_value = value
        # 处理离散值属性
        else:
            attri_values = set(attri_values)
            for value in attri_values:
                sub_dataset = splitDataset(dataset, index, value, is_continuous)
                p = len(sub_dataset)/float(len(dataset))
                new_ent += p * Ent(sub_dataset)
            gain = ent_D - new_ent
            if gain > max_gain:
                max_gain = gain
                attri_to_split = index
                split_value = None
    return attri_to_split, split_value


def majorClass(c_list):
    classCount = {}
    for vote in c_list:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 返回数组
    return sortedClassCount[0][0]


def decisionTree(dataset, attributes, all_dataset, all_attributes):
    class_list = [e[-1] for e in dataset]
    # 如果样本dataset全部属于同一类别c"是"或者"否"，返回c类叶节点
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 样本dataset 属性值相同，返回类别标签最多的类别
    # if all(len(count(dataset,attributes[i]))==1 for i in range(0,len(attributes)-1)):
    #     return majorClass(dataset)
    index, value = attriToSplit(dataset)
    attri_tosplit = all_attributes[index]

    mytree = {attri_tosplit: {}}

    # 获取划分属性attri的所有取值,由于随着迭代递归，子集中的样本减少，可能会丢失属性值
    # best_index = all_attributes.index(attri_tosplit)
    attri_values = set([e[index] for e in all_dataset])

    # 离散值
    if value == None:
        del(attributes[index])
        sub_attributes = attributes  # 子属性集中去除已划分的属性
        a_values = set([e[index] for e in dataset])
        # if a_values == attri_values:
        for val in a_values:
            mytree[attri_tosplit][val] = decisionTree(splitDataset(dataset, index, val, False),
                                                      sub_attributes,all_dataset, all_attributes)
    # 连续值
    else:
        sub_attributes = attributes
        mytree[attri_tosplit]['<='+str(value)] = decisionTree(splitDataset(dataset, index, value, False, 0),
                                                  sub_attributes,all_dataset, all_attributes)
        mytree[attri_tosplit]['>' + str(value)] = decisionTree(splitDataset(dataset, index, value, False, 1),
                                                                sub_attributes,all_dataset, all_attributes)

    return mytree





if __name__ == '__main__':
    file_path = 'data\watermelon3_0_Ch.csv'
    data, attributes = readData(file_path)
    all_dataset = data
    all_attributes = attributes
    mytree = decisionTree(data, attributes, all_dataset, all_attributes)
    print(mytree)
