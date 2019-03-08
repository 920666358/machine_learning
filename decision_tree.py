import numpy as np
import csv
from collections import defaultdict, Counter

# 读取数据
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
 
# 计算当前输入数据集的信息熵Ent
def Ent(dataset):
    labels = [vec[-1] for vec in dataset]
    labels_count = map(lambda label : label[1], Counter(labels).most_common())
    labels_p = map(lambda c : c / len(dataset), labels_count)
    labels_ent = map(lambda p : p*np.log2(p), labels_p)
    return -1 * sum(labels_ent)

# 基于某个属性 划分数据集
# 1.分类函数
split_funcs = {
    False : {
        0 : (lambda x, std_value : str(x) == str(std_value)),
    },
    True  : {
        0 : (lambda x, std_value : float(x) <= float(std_value)),
        1 : (lambda x, std_value : float(x) >  float(std_value)),
    }
}
# 2.划分
def splitDataset(dataset, index, value, is_continuous, part=0):
    sub_dataset = [vec for vec in dataset if split_funcs[is_continuous][part](vec[index], value)]
    return sub_dataset

def isDigitalStr(a):
    try :
        float(a)
        return True
    except :
        return False

def isContinuous(a) :
    return all(map(a, isDigitalStr))

def getCandidateValue(attr_list):
    if isContinuous(attr_list) :
        attr_list.sort()
        return set((attr_list[i]+attr_list[i+1])/2 for i in range(len(attr_list) -1))
    else :
        return set(attr_list)

def getPart(attr_list):
    if isContinuous(attr_list) :
        return [0,1]
    else :
        return [0]
def getSubDatasetsContinuous(dataset, index, attri_values) :
    split_ent_values = defaultdict(int)
    sub_datasets = defaultdict(list)
    for value in attri_values :
        for p in [0,1] :
            sub = splitDataset(dataset, index, value, True, p)
            split_ent_values[value] += Ent(sub)*(len(sub)/len(dataset))
            sub_datasets[value].append(sub)
    best_split_value = min(split_ent_values, split_ent_values.get)
    return best_split_value, sub_datasets[best_split_value]
    
def getSubDatasets(dataset, index) :
    attri_values = getCandidateValue( [e[index] for e in dataset] )  # 获取某属性的所有值的列表,不包括重复值
    sub_datasets = []
    split_value = None
    if isContinuous(attri_values) :
        split_value, sub_datasets = getSubDatasetsContinuous(dataset, index, attri_values)
    else :
        for v in set(attri_values) :
            sub_datasets.append(splitDataset(dataset, index, value, False))
    return split_value,sub_datasets

def attriToSplit2(dataset):
    candidate_split_index = dict()
    for index in range(len(dataset[0])-1):  # 最后一个值为y标签，而非属性值
        # 每一轮循环，处理一个属性
        split_value, sub_datasets = getSubDatasets(dataset, index)
        ent_list = map(lambda sub:Ent(sub)*(len(sub)/len(dataset)), sub_datasets)
        new_ent = sum(ent_list)
        candidate_split_index[index] = (new_ent, split_value)
    attri_to_split = min(candidate_split_index, lambda k: candidate_split_index[k][0]) # 返回ent最小的对应index
    split_value = candidate_split_index[attri_to_split][1] # 获取分割值，离散属性则为None

    # print('attri_to_split = {}'.format(attri_to_split))
    # print('split_value = {}'.format(split_value))
    return attri_to_split, split_value


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
    # print('attri_to_split = {}'.format(attri_to_split))
    # print('split_value = {}'.format(split_value))
    return attri_to_split, split_value


def count(dataset, index):
    num = {}
    for vec in dataset:
        if vec[index] in num:
            num[vec[index]] += 1
        else:
            num[vec[index]] = 1
    return num


def decisionTree(dataset, attributes):
    class_list = [e[-1] for e in dataset]

    # 如果样本dataset全部属于同一类别c"是"或者"否"，返回c类叶节点
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 样本dataset 属性值相同，返回类别标签最多的类别
    if all(len(count(dataset,i)) == 1 for i in range(0,len(dataset)-1)):
        return max(class_list, key=class_list.count)

    index, value = attriToSplit(dataset)  # index=1, value=None
    attri_tosplit = attributes[index]   # 根蒂
    mytree = {attri_tosplit: {}}
    # 连续值or离散值
    if value is None:
        attri_values = set(vec[index] for vec in dataset)
        for val in attri_values:
            mytree[attri_tosplit][val] = decisionTree(splitDataset(dataset, index, val, False),attributes)
    else:
        mytree[attri_tosplit]['<='+str(value)] = decisionTree(splitDataset(dataset, index, value, True, 0), attributes)
        mytree[attri_tosplit]['>'+str(value)] = decisionTree(splitDataset(dataset, index, value, True, 1), attributes)
    return mytree


def treeTest(tree, testdata, attibutes, predi=[]):
    index = attibutes.index(list(tree)[0])
    for vec in testdata:
        sub_tree = tree[list(tree)[0]][vec[index]]
        # print(sub_tree)
        if sub_tree in ['是','否']:
            predi.append(sub_tree)
            continue

        index_2 = attibutes.index(list(sub_tree)[0])
        attri_v = list(sub_tree[list(sub_tree)[0]].keys())
        if all(c in '0123456789.-' for c in vec[index_2]):
            split_value = float(attri_v[0][2:])
            ss_tree = sub_tree[list(sub_tree)[0]]   # {'<=0.3815': '否', '>0.3815': '是'}

            for key in ss_tree.keys():
                if key[0] == '<' and (float(vec[index_2]) <= split_value):
                    sss_tree = ss_tree[key]
                    if sss_tree in ['是', '否']:
                        predi.append(sss_tree)
                    else:
                        treeTest(sss_tree, [vec], attibutes, predi)
                    break
                elif key[0] == '>'and (float(vec[index_2]) > split_value):
                    sss_tree = ss_tree[key]
                    if sss_tree in ['是', '否']:
                        predi.append(sss_tree)
                    else:
                        treeTest(sss_tree, [vec], attibutes, predi)
                    break

        else:
            treeTest(sub_tree, [vec], attibutes, predi)
    return predi


def accuracy(predi, true_lables):
    count = 0
    for i in range(len(true_lables)):
        if predi[i] == true_lables[i]:
            count += 1
    return count/len(true_lables)


if __name__ == '__main__':
    file_path = 'data\watermelon3_0_Ch.csv'
    data, attributes = readData(file_path)
    mytree = decisionTree(data, attributes)
    print(mytree)

    testdata = [['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '0.403', '0.237'],
                ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '0.657', '0.198']]
    true_lables = ['是', '否']

    result = treeTest(mytree, testdata, attributes)
    print(result)

    a = accuracy(result, true_lables)
    print(a)
