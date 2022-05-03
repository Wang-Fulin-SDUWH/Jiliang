import pandas as pd
import numpy as np
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_white
from statsmodels.graphics.tsaplots import plot_acf

data=pd.read_excel('Data.xlsx')
data=data.values
X1=[]
X1_sq=[]
X6=[]
X6_sq=[]
X1X6=[]
Y=[]
res=[]
for i in range(len(data)):
    X1.append(np.log(data[i][2]))
    X6.append(data[i][7])
    X1_sq.append(np.log(data[i][2])**2)
    X6_sq.append(data[i][7]**2)
    X1X6.append(np.log(data[i][2])*data[i][7])
    Y.append(np.log(data[i][1]))
X_intercept=np.ones(len(data))
X1=np.array(X1)
X6=np.array(X6)
X=np.column_stack((X_intercept,X1,X6,X1_sq,X6_sq,X1X6)) # 添加截距
Y=np.array(Y)

for i in range(len(Y)):
    res.append((0.9054*X1[i]+0.0012*X6[i]-0.2408-Y[i])**2)
#res:残差平方数组

res=np.array(res)
model=sm.OLS(res,X)
results6=model.fit()
print(results6.summary())

res_ns=[]
for i in range(len(Y)):
    res_ns.append((0.9054*X1[i]+0.0012*X6[i]-0.2408-Y[i]))
res1_ns=res_ns[1:]
res_ns.pop(-1)
res_ns=np.array(res_ns)
res1_ns=np.array(res1_ns)
fz=0
print(len(res_ns))
print(len(res1_ns))
for i in range(len(res_ns)):
    fz+=res_ns[i]*res1_ns[i]
fm=sum(res)
print(fz)
print(fm)
print(fz/fm)
print('DW:',2*(1-fz/fm))