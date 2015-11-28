# coding: utf-8
import numpy as np
import sys

def read_data(filename):
    data_list = [] 
    with open(filename) as file_r:
        for line in file_r:
            vec = map(float, line.strip().split(','))
            data_list.append(vec)
    return data_list

def cal_set_info(dataset):
    '''计算一个给定数据集的信息熵'''
    data_count = {}
    info_d = 0.0

    for d in dataset:
        if d[-1] in data_count:
            data_count[d[-1]] += 1
        else:
            data_count[d[-1]] = 1

    for _, val in data_count.items():
        info_d -= (float(val) / len(dataset))  \
                * np.log2(float(val) / len(dataset))

    return info_d

def get_disc_val(dataset):
    '''
    获取所有离散属性的可能值
    '''
    dis_val = {}
    for idx, t in enumerate(dataset[0]):
        if t == 1.0:
            val = set()
            for d in dataset[1:]:
               val.add(d[idx])
            dis_val[idx] = list(val)
    return dis_val


def get_cls_from_data(dataset):
    data_count = {}
    
    for d in dataset:
        if d[-1] in data_count:
            data_count[d[-1]] += 1
        else:
            data_count[d[-1]] = 1
    
    key, max_val = 0, sys.float_info.min
    for k, v in data_count.items():
        if v > max_val:
            key, max_val = k, v

    return key

def binary_sp(data, border, index):
    left = {}
    right = {}
    for d in data:
        if d[index] < border:
            if d[-1] in left:
                left[d[-1]] += 1
            else:
                left[d[-1]] = 1 
        else:
            if d[-1] in right:
                right[d[-1]] += 1
            else:
                right[d[-1]] = 1
    return left, right


def cal_gain_ratio(stt_dict, data):
    '''
    根据统计结果计算增益率
    '''
    info_gain, info_measure = 0.0, 0.0
    total_info = cal_set_info(data)

    for _, v in stt_dict.items():
        total_d = sum(v.values())
        if total_d == 0:
            continue
        elif total_d == len(data):      #数据集对应的属性值都相同
            return 0, -1
        for _, i in v.items():
            if i == 0:
                continue
            info_gain -= (float(total_d) / len(data)) * \
                    (float(i) / total_d) * np.log2(float(i) / total_d)
        info_measure -= float(total_d) / len(data) * \
                np.log2(float(total_d) / len(data))

    return total_info - info_gain, info_measure


def check_purity(dataset):
    '''
    查看一个数据集是否只包含一个标签
    '''
    tmp_cls = dataset[0][-1]
    for d in dataset:
        if d[-1] != tmp_cls:
            return 0
    return 1


def check_accurcy(datasets, predict_cls):
    error_num = 0
    cls = []
    for d, c in zip(datasets, predict_cls):
        cls.append(d[-1])
        if d[-1] != c:
            error_num += 1
    print cls
    return 1 - float(error_num)/len(predict_cls)
        

if __name__ == '__main__':
    dataset = read_data("breast-cancer-assignment5.txt");
    print dataset[0]
    info_d = cal_set_info(dataset[1:])
    print info_d
    disc_val = get_disc_val(dataset)
    print disc_val
