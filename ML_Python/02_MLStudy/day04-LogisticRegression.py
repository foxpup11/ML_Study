# Author:司震
# Create: 2024/4/10
# Description: 第4天练习：逻辑回归
from sklearn.datasets import  load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#读取数据
X,y = load_iris(return_X_y=True)
X = X[:,[2,3]]
# 用分层抽样分割训练集和测试集
sss = StratifiedShuffleSplit(n_splits=1,test_size=.3,random_state=10)
for train_index,test_index in sss.split(X,y):
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
#建立模型 逻辑回归分类器
clf1 = LogisticRegression(penalty='l1',solver='liblinear',multi_class='ovr',C=2.0,max_iter=1000)
clf2 = LogisticRegression(penalty='l2',solver='lbfgs',multi_class='auto')
clf3 = LogisticRegression(penalty='l2',solver='saga',multi_class='multinomial',max_iter=1000)
clf4 = LogisticRegression(penalty='elasticnet',solver='saga',multi_class='auto',l1_ratio=0.95,max_iter=10000)
gs = gridspec.GridSpec(2,2)
labels =['penalty=11 solver=liblinear \nmulti_class=ovr C=2.0,max_iter=1000',
         'penalty=12 solver=lbfgs',
         'penalty=12 solver=lbfgs max_iter=1000',
         'penalty=elasticnet solver= saga\nll_ratio=0.95 max_iter-10000']
fig = plt.figure(figsize=(10,6),dpi=128)
for clf,label,grd in zip([clf1,clf2,clf3,clf4],labels,itertools.product([0,1],repeat=2)):
    clf.fit(X_train,y_train)#训练模型
    y_pred = clf.predict(X_test) #预测
    print('Accuracy:',accuracy_score(y_test,y_pred))
    #绘图
    ax = plt.subplot(gs[grd[0],grd[1]])
    fig= plot_decision_regions(X_test,y_test,clf,legend=2)
    plt.title(label)
    plt.text(1,2,'acc:'+str(clf.score(X_test,y_test))[:5],fontsize = 13)
    plt.xlabel('petal length(cm)')
    plt.ylabel('patal width(cm)')
plt.tight_layout()
plt.show()
