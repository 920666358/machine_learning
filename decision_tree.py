import numpy as np
import csv
from collections import defaultdict, Counter
from pprint import pprint

class Tree:
    def __init__(self, name):
        self.node_name = name
        self.rules = dict()
        self.datatype = '' # continues / discrete

    def GetRule(self, value) :
        if isDigitalStr(value) :
            for key in self.rules :
                try :
                    if(eval(str(value) + key)): # 将值代入表达式得到字符串，然后将字符串转换成python语句
                        return key
                except:
                    raise
        else :
            return value
    
    def search(self, test_data):
        if not isinstance(test_data, dict) :
            raise Exception("Error test_data format:expect dict type, but {} given".format(type(test_data)))
        if self.node_name not in test_data :
            print('The tree({}) is not suitable for test_data'.format(self.mode_name))
            return None
        value = test_data[self.node_name]
        rule = self.GetRule(value)
        if rule in self.rules :
            #print(type(self.rules[rule]))
            if isinstance(self.rules[rule], self.__class__) :
                return self.rules[rule].search(test_data)
            else :
                return self.rules[rule]
        else :
            print('No rules found for test_data:{}'.format(self.node_name))
            return None
    
    def __getitem__(self, key):
        return self.rules[key]

    def __setitem__(self, key, value):
        self.rules[key] = value

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self): # 自定义打印
        ss = '[{}]:\n'.format(self.node_name)
        for r in self.rules :
            ss += '    {}:\n'.format(r)
            for line in str(self.rules[r]).split('\n') :
                ss += '        {}\n'.format(line)
        return ss

# 读取数据
def readData(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        attributes = next(reader)[1:9]
        dataset = []
        for line in reader:
            dataset.append(line[1:10])
    # 返回数据集列表及属性列表print('dataset = {}'.format(dataset))
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

# 判断某种属性的数据类型：连续、离散
def isDigitalStr(a):
    try :
        float(a)
        return True
    except :
        return False

def isContinuous(a) :
    return all(map(isDigitalStr, a))

def getAttrType(attri_values) :
    if isContinuous(attri_values) :
        return 'continus'
    else :
        return 'discrete'

# 获取连续属性的所有候选分割点
def getCandidateSplitValue(attr_list):
    attr_list.sort()
    return set((float(attr_list[i])+float(attr_list[i+1]))/2 for i in range(len(attr_list) -1))

def getSubDatasetsContinuous(dataset, index, attri_values) :
    split_ent_values = defaultdict(int)
    sub_datasets = defaultdict(dict)
    patterns = { 0 : "<={}", 1 : ">{}" }
    for value in getCandidateSplitValue(list(attri_values)) :
        for p in patterns :
            sub = splitDataset(dataset, index, value, True, p)
            split_ent_values[value] += Ent(sub)*(len(sub)/len(dataset))
            label = patterns[p].format(value)
            sub_datasets[value][label] = sub
    best_split_value = min(split_ent_values, key = split_ent_values.get) #选择信息增益最大的，也即p*ent最小的
    return best_split_value, sub_datasets[best_split_value]

def getSubDatasetsDiscrete(dataset, index, attri_values) :
    sub_dataset = dict()
    for value in attri_values :
        sub_dataset[value] = splitDataset(dataset, index, value, False)
    return None, sub_dataset

# 按某种属性分割出子集
# 1.连续属性：计算出候选分割点，从候选分割点中选出信息熵最低的一个，并据此分割成两个子集
# 2.离散属性：按照离散属性的所有取值划分出子集
subset_funcs = {
    'continus' : getSubDatasetsContinuous,
    'discrete' : getSubDatasetsDiscrete,
}
def getSubDatasets(dataset, index) :
    attri_values = set(e[index] for e in dataset)  # 获取某属性的所有值的列表
    return subset_funcs[getAttrType(attri_values)](dataset, index, attri_values)

def attriToSplit(dataset):
    candidate_split_index = dict() # index:信息熵和,分割值,子集字典
    for index in range(len(dataset[0])-1):  # 最后一个值为y标签，而非属性值
        # 每一轮循环，处理一个属性
        split_value, sub_datasets = getSubDatasets(dataset, index) # 获取该属性划分出的所有子集
        ent_list = map(lambda sub:Ent(sub_datasets[sub])*(len(sub_datasets[sub])/len(dataset)), sub_datasets) # 计算出每个子集的信息熵*比重
        candidate_split_index[index] = (sum(ent_list), split_value, sub_datasets) # 存储每种属性的信息熵之和
    attri_to_split = min(candidate_split_index, key = lambda k: candidate_split_index[k][0]) # 找到信息熵之和最小的属性，也即信息增益最大
    sub_sets = candidate_split_index[attri_to_split][-1] # 获取连续属性分割值，离散属性则为None

    return attri_to_split, sub_sets

def count(dataset, index):
    value_list = [vec[index] for vec in dataset]
    num = dict(Counter(value_list).most_common()) # most_common返回(value,count)的元组列表，可以直接解析成字典
    return num

def decisionTree(dataset, attributes):
    class_list = [e[-1] for e in dataset]

    # 如果样本dataset全部属于同一类别c"是"或者"否"，返回c类叶节点
    if len(set(class_list)) == 1 :
        return class_list[0]

    # 样本dataset 属性值相同，返回类别标签最多的类别
    if all(len(count(dataset,i)) == 1 for i in range(0,len(dataset)-1)):
        return max(class_list, key=class_list.count)

    index, sub_datasets = attriToSplit(dataset)  # index=1, sub_datasets : {分割label:子集}
    attri_tosplit = attributes[index]   # 根蒂
    mytree = Tree(attri_tosplit)
    # 连续值or离散值统一处理
    for val in sub_datasets :
        mytree[val] = decisionTree(sub_datasets[val], attributes)

    return mytree

def treeTest(tree, testdata, attibutes, predi=[]):
    for vec in testdata :
        format_data = dict(zip(attibutes, vec))
        predi.append(tree.search(format_data))
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
