import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

data=pd.read_excel('Data.xlsx')
data=data.values
X1=[]
X2=[]
X3=[]
X4=[]
X5=[]
X6=[]
Y=[]
for i in range(len(data)):
    X1.append(data[i][2])
    X2.append(data[i][3])
    X3.append(data[i][4])
    X4.append(data[i][5])
    X5.append(data[i][6])
    X6.append(data[i][7])
    Y.append(data[i][1])
X_intercept=np.ones(len(data))
X1=np.array(X1)
X2=np.array(X2)
X3=np.array(X3)
X4=np.array(X4)
X5=np.array(X5)
X6=np.array(X6)
X=np.column_stack((X_intercept,X1,X2,X3,X4,X5,X6)) # 添加截距
Y=np.array(Y)
model=sm.OLS(Y,X)
results=model.fit()
print(results.summary())

X_pd=pd.DataFrame({'X1':X1,'X2':X2,'X3':X3,'X4':X4,'X5':X5,'X6':X6})
#print('解释变量之间的相关系数：\n',X_pd.corr())

for i in range(1,7):
    print('第'+str(i)+'个变量:',variance_inflation_factor(X,i))
