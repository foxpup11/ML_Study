# Author:司震
# Create: 2024/4/13
# Description: 第6天练习：贝叶斯方法
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  cross_val_score
from sklearn.datasets import load_breast_cancer



#利用高斯朴素贝叶斯方法进行鸢尾花分类
# #加载鸢尾花数据集
# X,y = load_iris(return_X_y=True)
# #构建高斯朴素贝叶斯
# gnb = GaussianNB()
# #目标选择的特征索引
# feature_groups = [[0,2],[0,3],[1,2],[1,3]]
# #获取属性列表
# feature_name = load_iris().feature_names
# #构造子图
# fig,axarr = plt.subplots(2,2,figsize=(10,8))
# for feature_group ,ax in zip(feature_groups,axarr.flat):
#     #根据目标属性进行数据划分
#     X_train,X_test,y_train,y_test = train_test_split(X[:,feature_group],y,test_size=0.3,random_state=0)
#     #拟合训练集，预测测试集
#     y_pred = gnb.fit(X_train,y_train).predict(X_test)
#     #计算准确度
#     acc = round(accuracy_score(y_test,y_pred),3)
#     #绘制分类边界
#     plot_decision_regions(X_train,y_train,gnb,legend=2,ax=ax)
#     #设置X轴Y轴标签
#     ax.set_xlabel('{}'.format(feature_name[feature_group[0]]))
#     ax.set_ylabel('{}'.format(feature_name[feature_group[1]]))
#     ax.set_title('acc:{}'.format(str(acc)))
# plt.tight_layout()
# plt.show()

# 采用三种朴素贝叶斯方法分类器进行鸢尾花分类
# X,y=load_iris().data,load_iris().target
# gn1 = GaussianNB()#高斯朴素贝叶斯方法
# gn2 = MultinomialNB()#多项式分布朴素贝叶斯方法
# gn3 = BernoulliNB()#伯努利朴素贝叶斯方法
# for model in [gn1,gn2,gn3]:
#     scores = cross_val_score(model,X,y,cv=10,scoring='accuracy')
#     print('Accuracy:{:.4f}'.format(scores.mean()))


#利用朴素贝叶斯方法对威斯康星州乳腺癌数据集进行分类
# X,y = load_breast_cancer().data,load_breast_cancer().target
# nb1=GaussianNB()
# nb2=MultinomialNB()
# nb3=BernoulliNB()
# for model in [nb1,nb2,nb3]:
#     scores = cross_val_score(model,X,y,cv=10,scoring='accuracy')
#     print('Accuracy:{:.4f}'.format(scores.mean()))






