# Author:司震
# Create: 2024/4/12
# Description: 第5天练习：决策树
from sklearn.datasets import load_iris
from  sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from  sklearn.datasets import load_digits

#基于决策树的鸢尾花分类
#导入数据
# iris = load_iris()
# X = iris['data']
# y = iris['target']
# feature_names = iris['feature_names']#获取数据的属性名
# target_names = iris['target_names']#获取数据的类别名
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=11)
# #分割数据
# clf_entropy = DecisionTreeClassifier(criterion='entropy')#构造决策树
# clf_entropy.fit(X_train,y_train)#训练决策树
# print(clf_entropy.score(X_test,y_test))#计算测试集上的分类准确率并输出
# fig = plt.figure(dpi=200)#指定图像的质量
# #显示决策树
# a =plot_tree(clf_entropy,#要绘制的决策树对象
#             feature_names=feature_names,#每个特征的名称
#             class_names=target_names )#每个类别的名称
# #plt.savefig(r'C:\Users\Administrator\Desktop\1.png')#保存在桌面
# plt.show()

#基于决策树的手写字符分类
# X,y = load_digits(return_X_y=True)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#分割数据
# acc_entropy = []
# acc_gini = []
# for i in range(1,20):
#     clf_entropy = DecisionTreeClassifier(criterion='entropy',max_depth=i)
#     clf_gini = DecisionTreeClassifier(criterion='gini',max_depth=i)
#     clf_entropy.fit(X_train,y_train)#训练entropy决策树
#     clf_gini.fit(X_train,y_train)#训练gini决策树
#     acc_entropy.append(clf_entropy.score(X_test,y_test))#保留entropy树准确率
#     acc_gini.append(clf_gini.score(X_test,y_test))#保留gini树准确率
# plt.plot(range(1,20),acc_entropy,'r--*',range(1,20),acc_gini,'b-.')
# plt.xlabel('max depth of decision tree')
# plt.ylabel('accuracy')
# plt.legend(['entropy','gini'])
# plt.show()


