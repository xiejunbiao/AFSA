#!-*-coding:utf-8-*-
from sko.AFSA import AFSA
import os
import sys
# pathDir = os.path.dirname(__file__)
# curPath = os.path.abspath(pathDir)
# rootPath = os.path.split(curPath)[0]
# sys.path.append(os.path.split(rootPath)[0])
# sys.path.append(rootPath)
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer #特征转换器
from sklearn.tree import DecisionTreeClassifier
import inspect
from decimal import Decimal
from decimal import getcontext
from copy import deepcopy
from sklearn.metrics import classification_report
getcontext().prec = 16


# 集合求交
def set_and(set1, set2):
    return list(set(set1).intersection(set(set2)))


# 集合求并
def set_union(set1, set2):
    return list(set(set1).union(set(set2)))


# 集合求差
def set_dif(set1, set2):

    # 有顺序
    return [i for i in set1 if i not in set2]


def sigmoid(x):
    # 直接返回sigmoid函数
    return 1. / (1. + np.exp(-x))


def _coding_(x, rand):
    result = []
    for i in x:
        if sigmoid(i) >= rand:
            result.append(1)
        else:
            result.append(0)
    return result


def check_dir_exist(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def decision_tree(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)  # 将数据进行分割
    vec = DictVectorizer(sparse=False)
    x_train = vec.fit_transform(x_train.to_dict(orient='record'))  # 对训练数据的特征进行提取
    x_test = vec.transform(x_test.to_dict(orient='record'))  # 对测试数据的特征进行提取
    # 转换特征后，凡是类别型型的特征都单独独成剥离出来，独成一列特征，数值型的则不变
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    Accracy = dtc.score(x_test, y_test)
    print('Accracy:', Accracy)
    # y_predict = dtc.predict(x_test)
    # print(classification_report(y_predict, y_test, target_names=['died', 'servived']))
    return Accracy


class FuncRough(object):
    def __init__(self, data):
        """

        :param data: 数据集其中类型为Dataframe, 最后一列为类别
        """
        self.u = data
        self.n, self.dim_num = self.u.shape
        self.d = self._d_()  # 获得数据最后一列的索引 表示决策属性
        self.c = self._c_()  # 获取除最后一列后的索引  表示条件属性
        # print(self.n)
        # print(self.dim_num)
        self._ind_d = self._ind_(self.d)  # self.ind_d 为一个字典

    def _c_(self):
        return pd.DataFrame({'c': self.u.dtypes.index.tolist()[:-1]})

    def _d_(self):
        return self.u.dtypes.index.tolist()[-1]

    def _ind_(self, b) -> dict:
        """
        :param b: 单个属性，或者多个属性列表
        :return: 返回按照单个属性或者多个属性进行划分的结果
        """
        ind_result = {}
        data_groupby_b = self.u.groupby(b)
        value_b_list = data_groupby_b.size().index.tolist()
        for value_b_each in value_b_list:
            ind_result[value_b_each] = data_groupby_b.get_group(value_b_each).index.to_list()
        return ind_result

    def b_(self, _code_0_1_list):
        """

        :param _code_0_1_list: 0、1串列表
        :return: 得到对应属性索引
        """
        _code_0_1_pd = pd.DataFrame({'code_0_1': _code_0_1_list})
        index = _code_0_1_pd[_code_0_1_pd['code_0_1'].isin([1])].index.to_list()
        return self.c[self.c.index.isin(index)]['c'].values.tolist()


    def _ind_b(self, _code_0_1_list):
        """

        :param _code_0_1_list: 0、1串列表
        :return: 得到对应属性的划分结果
        """
        column_index = self.b_(_code_0_1_list)
        ind_result = self._ind_(column_index)
        return ind_result

    def _up_down_like(self, k, dict_ind):
        y = self._ind_d[k]
        up_result = []
        down_result = []
        for key in dict_ind.keys():
            if len(set_and(dict_ind[key], y)) != 0:
                up_result = up_result + dict_ind[key]
            if len(set_dif(dict_ind[key], y)) == 0:
                down_result = down_result + dict_ind[key]
        return up_result, down_result

    def roughness(self, key, dict_ind):
        up_l, down_l = self._up_down_like(key, dict_ind)

        return len(set_dif(up_l, down_l))/len(up_l)

    def gk(self, dict_ind):
        result = 0
        for key in dict_ind.keys():
            result += len(dict_ind[key])**2
        return result/self.n**2

    def func_rough(self, b_1):
        """

        :param b_: 适应度函数接收到的参数为一串0-1编码
        :return:
        """
        b_1 = _coding_(b_1, 0.5)
        if np.sum(b_1) == 0:
            return 1
        relative_entropy = 0
        dict_ind = self._ind_b(b_1)
        gk = self.gk(dict_ind)
        for key in self._ind_d.keys():
            p = self.roughness(key, dict_ind)
            relative_entropy += p * np.log2(p + 1)
        return relative_entropy * gk


class AFSA_(object):
    def __init__(self, func, n_dim, size_pop=50, max_iter=300,
                 max_try_num=100, step=0.5, visual=0.3,
                 q=0.98, delta=0.5):
        self.control_print = None
        self.print_close()
        self.func = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.max_try_num = max_try_num  # 最大尝试捕食次数
        self.step = step  # 每一步的最大位移比例
        self.visual = visual  # 鱼的最大感知范围
        print(self.visual)
        self.q = q  # 鱼的感知范围衰减系数
        self.delta = delta  # 拥挤度阈值，越大越容易聚群和追尾

        self.X = 1 - 2*np.random.rand(self.size_pop, self.n_dim)
        # self.isprint_x_center(self.X)
        self.Y = np.array([self.func(x) for x in self.X])

        best_idx = self.Y.argmin()
        self.best_x, self.best_y = deepcopy(self.X[best_idx, :]), deepcopy(self.Y[best_idx])
        self.best_X, self.best_Y = self.best_x, self.best_y  # will be deprecated, use lowercase
        self.num = sum(_coding_(self.best_X, 0.5))

    def isprint_x_center(self, x):
        self.isprint('X', x)
        mean_x = x.mean(axis=0)
        self.isprint('mean_x', mean_x)
        distances = spatial.distance.cdist([mean_x,], self.X, metric='euclidean').reshape(-1)
        print('distances：%s' % distances)

    def is_more_num(self, x):
        return sum(_coding_(x, 0.5)) < self.num

    def move_to_target(self, idx_individual, x_target):
        '''
        move to target
        called by prey(), swarm(), follow()

        :param idx_individual:
        :param x_target:
        :return:
        '''
        x = self.X[idx_individual, :]

        # x_new = x + self.step * np.random.rand() * (x_target - x)
        detal = self.step * np.random.rand() * (x_target - x)
        x_new = x + detal

        fish_move_status = "第%d条人工鱼执行%s;从%s移动%s到%s：" % (idx_individual,
                                                        inspect.stack()[1][3],
                                                        x,
                                                        detal,
                                                        x_new)
        self.isprint('fish_move_status', fish_move_status)
        # x_new = x_target
        self.X[idx_individual, :] = x_new
        self.Y[idx_individual] = self.func(x_new)
        if Decimal(str(self.Y[idx_individual])) < Decimal(str(self.best_Y)):
            # print(1111111111111111111111)
            # print(self.Y[idx_individual])
            # print(self.Y[idx_individual])
            # print(self.Y[idx_individual]*100000000000 == self.Y[idx_individual]*100000000000)
            self.best_x = deepcopy(self.X[idx_individual, :])
            self.best_y = deepcopy(self.Y[idx_individual].copy())

    def move(self, idx_individual):
        '''
        randomly move to a point

        :param idx_individual:
        :return:
        '''
        r = 2 * np.random.rand(self.n_dim) - 1
        # r = 3 - 6 * np.random.rand(self.n_dim)
        x_new = self.X[idx_individual, :] + self.visual * r
        fish_move_status = "第%d条人工鱼执行%s;从%s移动%s到%s：" % (idx_individual,
                                                        inspect.stack()[1][3],
                                                        self.X[idx_individual, :],
                                                        self.visual * r,
                                                        x_new)
        self.isprint('fish_move_status', fish_move_status)
        self.X[idx_individual, :] = x_new
        self.Y[idx_individual] = self.func(x_new)
        if Decimal(str(self.Y[idx_individual])) < Decimal(str(self.best_Y)):
            # if self.Y[idx_individual] == self.best_Y:
            # and self.Y[self.best_Y] > 0.00000000001
            # print(1111111111111111111111)\
            # self.isprint('Y_best_Y', 'Y:%s, best_Y:%s' % (self.Y[idx_individual], self.best_Y))
            self.best_x = deepcopy(self.X[idx_individual, :])
            self.best_y = deepcopy(self.Y[idx_individual])

    def prey(self, idx_individual):
        '''
        prey
        :param idx_individual:
        :return:
        '''
        for try_num in range(self.max_try_num):
            r = 2 * np.random.rand(self.n_dim) - 1
            x_target = self.X[idx_individual, :] + self.visual * r
            if self.func(x_target) < self.Y[idx_individual]:  # 捕食成功
                self.move_to_target(idx_individual, x_target)
                return None
        # 捕食 max_try_num 次后仍不成功，就调用 move 算子
        self.move(idx_individual)

    def find_individual_in_vision(self, idx_individual):
        # 找出 idx_individual 这条鱼视线范围内的所有鱼
        # print('11111111111%s' % self.X[[idx_individual], :])
        distances = spatial.distance.cdist(self.X[[idx_individual], :], self.X, metric='euclidean').reshape(-1)
        if self.control_print['distances'] == 1:
            print("在执行：%s行为其他人工鱼距离当前人工鱼的距离%s" % (inspect.stack()[1][3], distances))
        # np.argwhere()返回非0的数组索引
        idx_individual_in_vision = np.argwhere((distances > 0) & (distances < self.visual))[:, 0]
        return idx_individual_in_vision

    def swarm(self, idx_individual):
        # 聚群行为
        idx_individual_in_vision = self.find_individual_in_vision(idx_individual)
        num_idx_individual_in_vision = len(idx_individual_in_vision)
        self.isprint('num_idx_individual_in_vision', "视线范围内的鱼数量为%d" % num_idx_individual_in_vision)
        if num_idx_individual_in_vision > 0:
            individual_in_vision = self.X[idx_individual_in_vision, :]
            center_individual_in_vision = individual_in_vision.mean(axis=0)
            center_y_in_vision = self.func(center_individual_in_vision)
            if center_y_in_vision * num_idx_individual_in_vision < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, center_individual_in_vision)
                return None
        self.prey(idx_individual)

    def follow(self, idx_individual):
        # 追尾行为
        idx_individual_in_vision = self.find_individual_in_vision(idx_individual)
        num_idx_individual_in_vision = len(idx_individual_in_vision)
        if num_idx_individual_in_vision > 0:
            individual_in_vision = self.X[idx_individual_in_vision, :]
            y_in_vision = np.array([self.func(x) for x in individual_in_vision])
            idx_target = y_in_vision.argmin()
            x_target = individual_in_vision[idx_target]
            y_target = y_in_vision[idx_target]
            if y_target * num_idx_individual_in_vision < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, x_target)
                return None
        self.prey(idx_individual)

    def run(self, max_iter=None):
        # 控制输出
        self.isprint('_coding_', _coding_(self.best_x, 0.5))
        best_y_and_mean = '第%d次迭代，最优粒度决策熵为:%s, 平均粒度决策熵：%s' % (1, self.best_y, self.Y.mean())
        self.isprint('best_y_and_mean', best_y_and_mean)

        message_trail = {}
        self.max_iter = max_iter or self.max_iter
        for epoch in range(self.max_iter - 1):

            for idx_individual in range(self.size_pop):
                self.swarm(idx_individual)
                self.follow(idx_individual)
            self.visual *= self.q
            message_trail[epoch] = {'best_x': self.best_x,
                                    'best_y': str(self.best_y),
                                    'mean': str(self.Y.mean())}
            # 控制输出
            self.isprint('_coding_', _coding_(self.best_x, 0.5))
            best_y_and_mean = '第%d次迭代，最优粒度决策熵为:%s, 平均粒度决策熵：%s' % (epoch+2, self.best_y, self.Y.mean())
            self.isprint('best_y_and_mean', best_y_and_mean)

        self.best_X, self.best_Y = self.best_x, self.best_y  # will be deprecated, use lowercase
        return self.best_x, self.best_y, message_trail

    def print_close(self):
        self.control_print = {
            'distances': 0,
            'fish_move_status': 0,
            'num_idx_individual_in_vision': 0,
            'best_y_and_mean': 1,
            '_coding_': 1,
            'Y_best_Y': 0,
            # 'X': 1,
            'X_radius': 1,
            'mean_x': 0
        }

    def isprint(self, key, str):
        """

        :param str: 需要输出的内容
        :return:
        """
        if key in self.control_print.keys() and self.control_print[key] == 1:
            print("%s: %s" % (key, str))

def main_2(file_path, logger=None):
    data_ = pd.read_csv(file_path)
    path_ = os.path.split(file_path)
    result_path = os.path.join(path_[0], 'result')
    check_dir_exist(result_path)
    red_data = os.path.join(result_path, 'step_result')
    check_dir_exist(red_data)
    n, m = data_.shape
    func_r = FuncRough(data=data_)
    afsa = AFSA_(func_r.func_rough, n_dim=m - 1, size_pop=15, max_iter=20,
                 max_try_num=10, step=0.8, visual=0.6 * np.sqrt(m - 1),
                 q=0.98, delta=0.5)
    best_x, best_y, message_trail = afsa.run()
    for key in message_trail.keys():
        code_0_1 = _coding_(message_trail[key]['best_x'], 0.5)  # 记录0-1编码
        sum_num_red = sum(code_0_1)
        red = func_r.b_(code_0_1)  # 具体的属性
        message_trail[key]['code_0_1'] = code_0_1
        message_trail[key]['red'] = red
        data_[red + [func_r.d]].to_csv(red_data+"//%s_%s.csv" % (path_[1], key))  # 根据具体属性获得固定数据
        accracy = decision_tree(data_[red], data_[func_r.d])  # 进行数据分类
        message_trail[key]['accracy'] = accracy  # 保存分类的准确率
        message_trail[key]['sum'] = sum_num_red  # 约简后的属性总个数
    pd_f = pd.DataFrame(message_trail).T  # 保存以上记录的信息（适应度、01编码，具体的约简属性、每一次迭代的准确率）
    pd_f.to_csv(os.path.join(result_path, path_[1]))

    print("best_x:%s" % best_x)
    print("best_y:%s" % best_y)


def main_0(__logger):
    data_path = "//home//afsa//AFSA//data//"
    file_list = ['breast5w.csv', 'Car.csv', 'dermatology.csv', 'heart.csv', 'MUSHROOM.csv',
                 'SPECT.csv', 'TIC_TAC_TOE2-9-2-958.csv', 'wdbc5w.csv', 'vote.csv']
    import os
    for file_name in file_list:
        data_path_t = os.path.join(data_path, file_name)
        __logger.info(data_path_t)
        main_2(data_path_t, __logger)
        __logger.info("%s 运行结束" % file_name)


if __name__ == '__main__':
    data_path = "E:\\研究生\\期刊\\谢俊标\\论文_构建中。。\\小论文-特征选择\\数据部分\\实验数据\\"

    file_list = ['breast5w.csv', 'Car.csv', 'dermatology.csv', 'heart.csv', 'MUSHROOM.csv',
                 'Nursery.csv', 'SPECT.csv', 'TIC_TAC_TOE2-9-2-958.csv', 'wdbc5w.csv', 'vote.csv']
    # file_list = ['wdbc5w.csv']
    # func_r = FuncRough(data_)
    # r = 2 * np.random.rand(m) - 1
    # print(r)
    # print(_coding_(r, 0.5))
    # print(func_r.func_rough(r))
    import os
    for file_name in file_list:
        data_path_t = os.path.join(data_path, file_name)
        print(data_path_t)
        main_2(data_path_t)
