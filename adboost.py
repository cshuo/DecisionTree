# coding: utf-8
import numpy as np

from C45 import DecisionTree
from utils import (
        read_data,
        get_disc_val,
        check_accurcy,
        fcv,
)


def adaboost(dataset, iterat_num):
    '''
    adaboost算法生成多个不同权重的分类器.
    '''
    weighted_classifier = []
    data_weight = [float(1) / len(dataset)] * len(dataset)
    err_rate, t = 1.0, 0

    while t < iterat_num:
        print "迭代: ", t

        if t != 0:          #按照权重采样数据集
            dataset_smp = sample_data(dataset, data_weight)
        else:
            dataset_smp = dataset

        #根据采集的数据样本生成决策树
        decisin_tree = DecisionTree(dataset_smp, AttrSet, DiscType)
        pred_res = get_pre_res(dataset, decisin_tree.classify(dataset))

        err_rate = wgh_err_rate(data_weight, pred_res)

        tree_weight =  np.log(1 / err_rate - 1) / 2
        weighted_classifier.append([tree_weight, decisin_tree])

        update_wgh(data_weight, pred_res, tree_weight)

        t += 1

    return weighted_classifier



def ada_classify(tran_data, test_data):
    '''
    使用训练的adaboost决策树对测试数据进行预测
    '''
    res_cls = []
    sub_tree_wh = []
    wh_classifier = adaboost(tran_data[1:],10)
    final_cls = []

    '''
    [   [权重，决策树],
        ...
    ]
    '''

    for wh_tree in wh_classifier:
        sub_tree_wh.append(wh_tree[0])
        res_cls.append(wh_tree[1].classify(test_data))

    clses_T = map(list, zip(*res_cls))

    for c in clses_T:
        vote_res = {}
        for i, wh in zip(c, sub_tree_wh):
            if i in vote_res:
                vote_res[i] += wh
            else:
                vote_res[i] = wh
        final_cls.append(max(vote_res, key=vote_res.get))

    print final_cls
    accurcy = check_accurcy(test_data, final_cls)
    return accurcy


def sample_data(dataset, data_wh):
    '''
    基于权重对数据进行采样,采样的大小为原数据集的大小
    '''
    data_index = range(len(dataset))
    sample_index = np.random.choice(data_index, len(dataset), p = data_wh)
    data_s = []
    for idx in sample_index:
        data_s.append(dataset[idx])
    return data_s


def wgh_err_rate(data_wh, pred_res):
    '''
    计算基于权重的错误率
    '''
    #print pred_res
    err_rt = 0.0
    for w, sign in zip(data_wh, pred_res):
        if sign == 0:
            err_rt += w
    return err_rt


def update_wgh(data_wh, pred_res, tree_wh):
    '''
    根据前一次迭代的分类器的权重，以及分类的结果来对数据集权重进行更新
    '''
    for idx, _ in enumerate(data_wh):
        if pred_res[idx] == 0:
            data_wh[idx] *= np.exp(tree_wh)
        else:
            data_wh[idx] *= np.exp(-tree_wh)

    total_wh = sum(data_wh)
    for idx, _ in enumerate(data_wh):
        data_wh[idx] /= total_wh
    data_wh[-1] += 1 - sum(data_wh)             #确保权重之和为1


def get_pre_res(dataset, res_cls):
    '''
    根据分类结果以及训练集本身的标签，对正确以及错误分类进行统计
    '''
    pre_statis = []
    for d, cls in zip(dataset, res_cls):
        if d[-1] == cls:
            pre_statis.append(1)
        else:
            pre_statis.append(0)
    return pre_statis

if __name__ == '__main__':
    dataset =  read_data("german-assignment5.txt")
    #dataset =  read_data("breast-cancer-assignment5.txt")
    DiscType = get_disc_val(dataset)
    AttrSet = range(len(dataset[0]))

    print fcv(dataset, ada_classify)
