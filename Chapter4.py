import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# 读取数据
# 第一问
data=pd.read_excel('Data.xlsx')
data=data.values
X2=[]
X3=[]
X4=[]
X5=[]
X6=[]
X7=[]
Y=[]
for i in range(len(data)):
    X2.append(data[i][2])
    X3.append(data[i][3])
    X4.append(data[i][4])
    X5.append(data[i][5])
    X6.append(data[i][6])
    X7.append(data[i][7])
    Y.append(data[i][1])
X_intercept=np.ones(len(data))
X2=np.array(X2)
X3=np.array(X3)
X4=np.array(X4)
X5=np.array(X5)
X6=np.array(X6)
X7=np.array(X7)
X=np.column_stack((X_intercept,X2,X3,X4,X5,X6,X7)) # 添加截距
Y=np.array(Y)
model=sm.OLS(Y,X)
results=model.fit()
print(results.summary())

# 第三问（1）
X_pd=pd.DataFrame({'X2':X2,'X3':X3,'X4':X4,'X5':X5,'X6':X6,'X7':X7})
print(X_pd.corr())

# # 第三问（2）
# Y_2=X2
# X_2=np.column_stack((X_intercept,X3,X4,X5))
# model_2=sm.OLS(Y_2,X_2)
# results_2=model_2.fit()
# print('*'*100)
# print('X2为被解释变量时：')
# print(results_2.summary())
# Y_3=X3
# X_3=np.column_stack((X_intercept,X2,X4,X5))
# model_3=sm.OLS(Y_3,X_3)
# results_3=model_3.fit()
# print('*'*100)
# print('X3为被解释变量时：')
# print(results_3.summary())
# Y_4=X4
# X_4=np.column_stack((X_intercept,X2,X3,X5))
# model_4=sm.OLS(Y_4,X_4)
# results_4=model_4.fit()
# print('*'*100)
# print('X4为被解释变量时：')
# print(results_4.summary())
# Y_5=X5
# X_5=np.column_stack((X_intercept,X2,X3,X4))
# model_5=sm.OLS(Y_5,X_5)
# results_5=model_5.fit()
# print('*'*100)
# print('X5为被解释变量时：')
# print(results_5.summary())