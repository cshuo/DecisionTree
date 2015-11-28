# coding: utf-8
import sys
from collections import defaultdict
from operator import itemgetter
import numpy as np


from utils import (
    read_data, 
    cal_set_info,
    get_disc_val,
    binary_sp,
    cal_gain_ratio,
    check_purity,
    check_accurcy,
    get_cls_from_data
)

class TreeNode(object):
    def __init__(self):
        self.cls = 0.0                  # class of the leaf node
        self.childNode = {}             # type also TreeNode, 属性值: TreeNode
        self.attr_type = 0              # 1 for categorical, 0 for numerical
        self.attr_index = -1            # 分裂数据集所使用属性的序号
        self.demark = 0.0               # 如果是连续属性，分界点


class DecisionTree(object):
    def __init__(self,dataset):
        self.dataset = dataset                      #包含第一行的属性类别指示
        self.root = TreeNode()
    

    def __construct_tree(self, cur_node, attr_list, data):
        '''
        递归构建决策树
        '''
        data_classified = {}
        max_gain_ratio, index = sys.float_info.min, -1
        num_border = 0.0

        for idx in attr_list:
            if self.dataset[0][idx] == 1.0:            #离散属性
                gain_r = self.disc_gain_rt(idx, data)
            else:                                      #数值属性
                gain_r, num_border = self.num_gain_rt(idx,data)

            if gain_r > max_gain_ratio:
                max_gain_ratio = gain_r
                index = idx

        if index == -1:                                #所有属性都不能满足条件
            cur_node.cls = get_cls_from_data(data)
            return

        cur_node.attr_index = index

        if index in DiscType: 
            cur_node.attr_type = 1

            #对数据进行分类
            for val in DiscType[index]:
                data_classified[val] = []
            for d in data:
                data_classified[d[index]].append(d)
            
        else:
            cur_node.attr_type = 0
            cur_node.demark = num_border
            
            data_classified[0] = []
            data_classified[1] = []
            for d in data:
                if d[index] < num_border:
                    data_classified[0].append(d)
                else:
                    data_classified[1].append(d)

        #print "子节点数据集"
        #for k,v in data_classified.items():
            #print ">>>>", k
            #print v

        if len(attr_list) == 1:                 #下一次递归属性集为空
            for k, v in data_classified.items():
                child_node = TreeNode()
                #属性值对应的数据集为空，则使用当前节点的数据集判断节点对应的分类
                if len(v) == 0:                 
                    child_node.cls = get_cls_from_data(data)
                else:
                    child_node.cls = get_cls_from_data(v)
                cur_node.childNode[k] = child_node
        else:
            attr_list.remove(index)
            for k, v in data_classified.items():
                child_node = TreeNode()
                if len(v) == 0:
                    child_node.cls = get_cls_from_data(data)
                elif check_purity(v) == 1:
                    child_node.cls = v[0][-1]           #随便取一个sample的标签
                else:
                    self.__construct_tree(child_node, attr_list, v)
                cur_node.childNode[k] = child_node
                    

    def construct_tree(self):
        '''
        决策树递归构建entrance
        '''
        init_attr_list = range(len(self.dataset[0])) 
        self.__construct_tree(self.root, init_attr_list, self.dataset[1:176])


    def disc_gain_rt(self, index, data):
        '''
        计算一个属性的信息增益
        '''
        statisc_dict = {}
        #info_gain, info_measure = 0.0, 0.0
        index_val = DiscType[index]

        total_info = cal_set_info(data)

        for val in index_val:
            statisc_dict[val] = {}

        for d in data:
            if d[-1] in statisc_dict[d[index]]:
                statisc_dict[d[index]][d[-1]] += 1
            else:
                statisc_dict[d[index]][d[-1]] = 1
        
        '''
        statisc_dict结构:
        {
            attr_value1:{yes:num1, no:num2},
            ...
            attr_value2:{yes:num1, no:num2},
        }
        '''
        info_gain, info_measure = cal_gain_ratio(statisc_dict, data)

        return -1 if info_measure == -1 else info_gain / info_measure


    def num_gain_rt(self, index, data):
        '''
        连续数值属性计算信息增益，先根据第index列排序，选取标签改变时对应的index列属性值，
        作为分界点，分别计算出每个分界点对应的信息增益，返回最大增益及其对应的分界点
        '''
        ctgs = set()
        sorted_data = sorted(data, key=itemgetter(index))
        cls = sorted_data[0][-1]

        #只选取便签改变时对应的属性值 
        for d in sorted_data:
            if d[-1] != cls:
                cls = d[-1]
                ctgs.add(d[index])

        
        max_gain, border, gain_ratio = sys.float_info.min, 0.0, -1.0
        for ctg in ctgs:
            statisc_dict = {}
            info_gain = 0.0
            '''
            结构为
            {
                'left': {yes: num1, no:num2}
                'right': {yes:num1, no:num2}
            }
            '''
            statisc_dict['left'], statisc_dict['right'] = binary_sp(data, ctg, index)
            
            info_gain, info_measure = cal_gain_ratio(statisc_dict, data)

            if info_measure == -1:
                continue
            if info_gain > max_gain:
                max_gain, border, gain_ratio = info_gain, ctg, info_gain / info_measure 

        return  gain_ratio, border
    

    def classify(self,dataset):
        predict_cls = []
        for d in dataset:
            predict_cls.append(self.__classify_data(d, self.root))
        return predict_cls


    def __classify_data(self, data, cur_node):
        if len(cur_node.childNode) == 0:
            return cur_node.cls
        else:
            criteria_val = data[cur_node.attr_index]
            if cur_node.attr_type == 1:            #离散属性
                next_node = cur_node.childNode[criteria_val]
                return self.__classify_data(data, next_node)
            else:
                if criteria_val < cur_node.demark:
                    next_node = cur_node.childNode[0]
                else:
                    next_node = cur_node.childNode[1]
                return self.__classify_data(data, next_node)


if __name__ == '__main__':
    #dataset =  read_data("test.txt")
    dataset =  read_data("breast-cancer-assignment5.txt")
    #dataset =  read_data("german-assignment5.txt")
    DiscType =  get_disc_val(dataset)
    decisin_tree = DecisionTree(dataset)
    decisin_tree.construct_tree()
    #decisin_tree.iter_tree()
    res_cls = decisin_tree.classify(dataset[176:278])
    print res_cls
    acc = check_accurcy(dataset[176:278], res_cls)
    print acc
