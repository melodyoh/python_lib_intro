# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 10:35:48 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing  
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_curve, roc_curve, auc,roc_auc_score

#numpy
print("\n-------------创建向量/矩阵-----------------\n")
#创建向量/矩阵
a = np.array([5, 10, 15, 20])
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("a=\n",a)
print("b=\n",b)
print("\n-------------输出/修改行列数-----------------\n")
#输出行列数
print("a.shape=",a.shape)
print("b.shape=",b.shape)
print("b.shape[0]=",b.shape[0])

#修改行列数
b.shape=4,3 #此处非转置
c=b.reshape((2,-1)) # -1，根据数组元素的个数自动计算长度
print("b=\n",b)
print("c=\n",c)

#此处b,c共享内存，修改任意一个将影响另外一个
c[0][0]=2
print("b=\n",b)
print("c=\n",c)

#转置
b=b.T
print("b=\n",b)


print("\n-------------指定数据类型创建-----------------\n")
#指定数据类型
d=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]],dtype=np.float)
print("d=\n",d)

d=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]],dtype=np.complex)
d=np.array([[1,0,3,4]],dtype=np.bool)

#转换数据类型
f=d.astype(np.int)
print("f=\n",f)

print("\n-------------使用函数创建-----------------\n")
#使用函数创建
#等差(指定公差)
a=np.arange(1,5,0.5)
print('a=',a)
#等差(指定个数)
b=np.linspace(1,10,10)
print('b=',b)

#endpoint指定是否包括终值
b=np.linspace(1,10,10, endpoint=False)
print('b=',b)

#等比
c=np.logspace(1,4,4,endpoint=True,base=2)
print('c=',c)
c= np.logspace(1,4,4, endpoint=False, base=2)
print('c=',c)

print("\n-------------存取-----------------\n")
a=np.arange(1,10)
print('a= ',a)
#获取某个元素
print('a[3]= ',a[3])
#获取第3-6个元素，左必右开
print('a[3:6]= ',a[3:6])
#省略开始下标，表示从0开始
print('a[:5]= ',a[:5])
#步长为2
print('a[1:9:2]= ',a[1:9:2])
#步长为-1，即翻转
print('a[::-1]= ',a[::-1])

#切片数据是原数组的一个视图，与原数组共享内容空间
b = a[2:5] 
b[0] = 200
print('b= ',b)
print('a= ',a)

print("\n-------------随机生成-----------------\n")
np.random.seed(0)
#均匀分布的随机数
a = np.random.rand(10)
#指定shape
b = np.random.rand(3,2)
print('a= ',a)
print('b= \n',b)

#正态分布的随机数, eg.N(3, 6.25)
c = 2.5 * np.random.randn(2, 4) + 3
print('c= ',c)

#0-4整数
d = np.random.randint(5, size=(2, 4))
print('d= \n',d)

#洗牌
e= np.arange(10)
print('e= ',e)
np.random.shuffle(e)
print('e= ',e)

print("\n-------------使用布尔数组存取-----------------\n")
#大于0.5的元素索引
print (a > 0.5)
#大于0.5的元素
b = a[a > 0.5]
print('b= ',b)
#将数组b中大于0.5的元素截取成0.5
b[b > 0.5] = 0.5
print('b= ',b)
#a不受影响
print('a= ',a)

print("\n-------------计算（点乘、求和…）-----------------\n")
A=np.array([[1,1],[0,1]])
B=np.array([[2,0],[3,4]])
print('A=\n ',A)
print('B=\n ',B)
#元素相乘element-wise product
A*B
print('A*B=\n ',A*B)
#点乘
A.dot(B)
np.dot(A,B)
print('A.dot(B)=\n ',A.dot(B))
print('np.dot(A,B)=\n ',np.dot(A,B))
#求和
B.sum(axis=1)
B.sum(axis=1,keepdims=True)
#max,min,sqrt,exp...

print("\n-------------meshgrid-----------------\n")
#meshgrid用于从数组a和b产生网格
u = np.linspace(-3, 3, 101)
x, y = np.meshgrid(u, u)

z = x*y*np.exp(-(x**2 + y**2)/2) / math.sqrt(2*math.pi)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0.1) 

print("\n-------------例子-----------------\n")
print(np.arange(0,60,10).reshape((-1,1))+np.arange(6))

#pandas
print("\n-------------Series-----------------\n")
a= pd.Series([4,7,-5,3])
print(a)
b= pd.Series([4,7,-5,3],index=['d','b','a','c'])
print(b)
print("\n-------------DataFrame-----------------\n")
d = {'state':['1','2'],'year':['a','b'],'pop':['x','y']}
print(d)
frame = pd.DataFrame(d)
print(frame)
#追加数据
frame2=pd.DataFrame([['z','3','c'],['x','4','d']],columns=['pop','state','year'])
frame.append(frame2,ignore_index=True)
#拼接数据
pd.concat([frame,frame2])

print("\n-------------基本用法-----------------\n")
#读取数据
data=pd.read_csv('student-por.csv',delimiter=";")
data.shape
#显示头/尾几行
data.head()
data.tail()
#显示列名/值
data.columns
data.values
print("\n-------------筛选、缺失值处理-----------------\n")
#筛选行/列
data.iloc[3:6]
data.iloc[:,3:6]
data.loc[:,["school", "age"]]
#条件筛选
data[data["G1"]<10]
#缺失值处理
data.fillna(value=0)
data.dropna(how="any")
data.isnull()

print("\n-------------排序-----------------\n")
#排序
data.sort_values("G1", ascending=False)
#统计并排序
s=pd.Series(data.loc[:,"Medu"])
s=s.value_counts()
s=s.sort_index(axis=0)

print("\n-------------算术运算、统计-----------------\n")
#数据描述
data.describe()
#算术运算
s=pd.Series(data.loc[:,"G1"])
s.mean()
data.loc[:,"G1"].mean()
data.mean(1)

#groupby统计
data.groupby(['sex', 'studytime'])['G1'].mean()

group1 = data.groupby('sex')
group1['G1','G2'].agg(['mean','sum'])

#数据透视表
pd.pivot_table(data, values='G1', index=['sex'],columns=['age'], aggfunc=np.mean)



print("\n-------------类别转换-----------------\n")
#类别转换
medu=data["Medu"].astype("category")
medu.cat.categories=["None","<4th grade","5th to 9th grade","secondary education","higher education"]
#转换成哑元
Fedu_dummies = pd.get_dummies(data["Fedu"], prefix='Fedu')
data=data.join(Fedu_dummies)


#matplotlib
print("\n-------------基本绘图-----------------\n")
x = np.linspace(start=-3, stop=3, num=1001, dtype=np.float)
x1=x.reshape(1,1001)
zero= np.zeros((1,1001))
y_relu=np.max(np.vstack((x1,zero)),axis=0)
y_logit=1/(1+np.exp(-x))
y_tanh=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
plt.figure(figsize=(8,6))
plt.ylim((-1, 1))
plt.plot(x,y_logit,'r-',label='LogisticLoss',linewidth=2)
plt.plot(x,y_tanh,'g-',label='TanhLoss',linewidth=2)
plt.plot(x,y_relu,'b-',label='ReluLoss',linewidth=2)
plt.title("Lossfunction",fontsize=18)
plt.grid()
plt.legend(loc='upperright')
plt.show()
#plt.savefig('1.png')

#x = np.linspace(start=-3, stop=3, num=1001, dtype=np.float)
#x1=x.reshape(1,1001)
#x1[x1<0]=0

print("\n-----直方图/箱图/条形图/...-----\n")
data=pd.read_csv('student-por.csv',delimiter=";")
df=pd.DataFrame(data)

plt.hist(df.loc[:,"G1"],bins=19)
plt.xlabel('Performance',fontsize=18)
plt.ylabel('Num of Students',fontsize=18)
plt.title('Histogram of {0}'.format('G1'),fontsize=18)
plt.show()

df.boxplot(column=["G1"],by="Medu")
plt.show()

s=pd.Series(df.loc[:,'sex'])
s=s.value_counts()
s=s.sort_index(axis=0)

s.plot(kind='barh')
plt.ylabel('SEX')
plt.show()

df.loc[:,['G1','G2','G3']].plot(kind='kde')
plt.show()

#kind=  line	线图
#	pie	饼图
#	bar	垂直条形图
#	barh	水平条形图
#	kde	核密度估计
#	hist	直方图
#	box	箱图

print("\n-------------数据预处理-------------\n")
#正则化
X = [[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]]
print(preprocessing.normalize(X, norm='l2'))
#preprocessing.normalize(X, norm='l1')

#特征二值化
binarizer=preprocessing.Binarizer(copy=True, threshold=0.0).fit(X)
print(binarizer.transform(X))
#OneHotEncoder
enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
print(enc.transform([[0, 1, 3]]).toarray())
#array([[0, 0, 3],
#       [1, 1, 0],
#       [0, 2, 1],
#       [1, 0, 2]])

#enc.fit(train_feature)  
#train_feature = enc.transform(train_feature).toarray()  
#test_feature = enc.transform(test_feature).toarray()  

#缺失值处理
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))                   

#from (X1, X2) to (1, X1, X2, X1^2, X1X2, X2^2)
X = np.arange(6).reshape(3, 2)
poly = preprocessing.PolynomialFeatures(2)
print(poly.fit_transform(X))                             

print("\n-------------训练数据-------------\n")
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
            ]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()]
    
figure = plt.figure(figsize=(27, 9))
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
h=0.02
i = 1

for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = preprocessing.StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    if ds_cnt == 0:
        ax.set_title("Input data")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
                                  
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)
        if ds_cnt == 0:
            ax.set_title(name)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()

