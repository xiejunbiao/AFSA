#coding=utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer #特征转换器
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree

# #1.数据获取
# titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# #print titanic.head()
# #print titanic.info()
# X = titanic[['pclass','age','sex']]  # 提取要分类的特征。一般可以通过最大熵原理进行特征选择
# y = titanic['survived']
# print(X.shape)   #(1313, 3)
# #print X.head()
# #print X['age']
#
# #2.数据预处理：训练集测试集分割，数据标准化
# # X['age'].fillna(X['age'].mean(),inplace=True)   # age只有633个，需补充，使用平均数或者中位数都是对模型偏离造成最小的策略
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)  # 将数据进行分割
#
# vec = DictVectorizer(sparse=False)
# X_train = vec.fit_transform(X_train.to_dict(orient='record'))   # 对训练数据的特征进行提取
# X_test = vec.transform(X_test.to_dict(orient='record'))         # 对测试数据的特征进行提取
# # 转换特征后，凡是类别型型的特征都单独独成剥离出来，独成一列特征，数值型的则不变
# print(vec.feature_names_)  # ['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']
#
# # 3.使用决策树对测试数据进行类别预测
# dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)
# y_predict = dtc.predict(X_test)
#
# # 4.获取结果报告
# print('Accracy:', dtc.score(X_test,y_test))
# print(classification_report(y_predict,y_test,target_names=['died','servived']))
#
# # 5.将生成的决策树保存
# with open("jueceshu.dot", 'w') as f:
#     f = tree.export_graphviz(dtc, out_file=f)


def DecisionTree(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)  # 将数据进行分割
    vec = DictVectorizer(sparse=False)
    x_train = vec.fit_transform(x_train.to_dict(orient='record'))  # 对训练数据的特征进行提取
    x_test = vec.transform(x_test.to_dict(orient='record'))  # 对测试数据的特征进行提取
    # 转换特征后，凡是类别型型的特征都单独独成剥离出来，独成一列特征，数值型的则不变
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    y_predict = dtc.predict(x_test)
    # print('Accracy:', dtc.score(x_test, y_test))
    print(classification_report(y_predict, y_test, target_names=['died', 'servived']))
    return dtc.score(x_test, y_test)
