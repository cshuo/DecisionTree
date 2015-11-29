# coding: utf-8
import numpy as np
import random
import collections

from C45 import DecisionTree
from utils import (
        read_data,
        get_disc_val,
        check_accurcy,
        fcv,
)

def random_fr(dataset):
    attrset = range(len(dataset[0]))
    attr_select_num = int(np.sqrt(len(attrset)) / 2 + 1)        #每次选择的属性数目
    forests = []
    
    while len(attrset) > 0:
        attr_s = []         #子树属性集
        data_s = []         #子树数据集
        for i in xrange(attr_select_num):
            if len(attrset) == 0:
                break
            attr = random.choice(attrset)
            attrset.remove(attr)
            attr_s.append(attr)
        for i in xrange(len(dataset)-1):
            data_s.append(random.choice(dataset[1:]))
        
        forests.append(DecisionTree(data_s, attr_s, DiscType))

    return forests

def rd_fr_classify(tran_data, test_data):
    forests = random_fr(tran_data)
    res_clses = []
    cls = []

    for tree in forests:
        res_clses.append(tree.classify(test_data))

    clses_T = map(list, zip(*res_clses))

    for c in clses_T:
        vote_cls = collections.Counter(c).most_common(1)[0][0]
        cls.append(vote_cls)

    accurcy = check_accurcy(test_data, cls)
    return accurcy




if __name__ == '__main__':
    #dataset =  read_data("breast-cancer-assignment5.txt")
    dataset =  read_data("german-assignment5.txt")
    DiscType = get_disc_val(dataset)
    #forests = random_fr(dataset)
    #accurcy = rd_fr_classify(dataset, dataset[1:])
    #print accurcy
    print fcv(dataset, rd_fr_classify)
    
