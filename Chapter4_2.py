import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# 读取数据
# 第一问
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
X_intercept=np.ones(len(data))
X2=np.array(X2)
X3=np.array(X3)
X4=np.array(X4)
X5=np.array(X5)
X=np.column_stack((X_intercept,X3,X4)) # 添加截距
Y=np.array(Y)
model=sm.OLS(Y,X)
results=model.fit()
print(results.summary())


Y_3=X3
X_3=np.column_stack((X_intercept,X4))
model_3=sm.OLS(Y_3,X_3)
results_3=model_3.fit()
print('*'*100)
print('X3为被解释变量时：')
print(results_3.summary())

X21=np.diff(X2)
X31=np.diff(X3)
X41=np.diff(X4)
X51=np.diff(X5)
Y_diff=np.diff(Y)
X_intercept=np.ones(len(data)-1)
X_diff=np.column_stack((X_intercept,X21,X31,X41,X51))
model_diff=sm.OLS(Y_diff,X_diff)
results_diff=model_diff.fit()
print('*'*100)
print('差分后：')
print(results_diff.summary())