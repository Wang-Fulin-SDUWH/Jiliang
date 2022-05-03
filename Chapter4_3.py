from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
clf = Ridge(alpha=1000)

data=pd.read_excel('Chapter4.xlsx')
data=data.values
X2=[]
X3=[]
X4=[]
X5=[]
Y=[]
for i in range(len(data)):
    X2.append(data[i][1])
    X3.append(data[i][2])
    X4.append(data[i][3])
    X5.append(data[i][4])
    Y.append(data[i][0])
X2=np.array(X2)
X3=np.array(X3)
X4=np.array(X4)
X5=np.array(X5)
X=np.column_stack((X2,X3,X4,X5)) # 添加截距
Y=np.array(Y)
clf.fit(X,Y)
print('coefs:',clf.coef_)
print('intercept:',clf.intercept_)
print('R2:',clf.score(X,Y))
print(clf.get_params())
