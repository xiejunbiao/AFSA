#!-*-coding:utf-8-*-
import os
import sys
pathDir = os.path.dirname(__file__)
curPath = os.path.abspath(pathDir)
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
sys.path.append(rootPath)
import pandas as pd
from feature_select.func_fit import FuncRough
from feature_select.AFSA_ import AFSA


def main(func):
    afsa = AFSA(func, n_dim=16, size_pop=15, max_iter=50,
                max_try_num=10, step=0.5, visual=0.3,
                q=0.98, delta=0.5)
    afsa.run()


if __name__ == '__main__':
    data_path = "E:\\研究生\\期刊\\谢俊标\\论文_构建中。。\\小论文-特征选择\\数据部分\\实验数据\\vote.csv"
    data_ = pd.read_csv(data_path)
    func_r = FuncRough(data=data_)
    main(func_r.func_rough)
