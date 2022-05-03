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


data=pd.read_excel('Data.xlsx')
data=data.values
X1=[]
X1_sq=[]
X6=[]
X6_sq=[]
X1X6=[]
Y=[]
res1=[]
res2=[]
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

for i in range(0,12):
    res1.append((0.9054*X1[i]+0.0012*X6[i]-0.2408-Y[i])**2)
for i in range(18,30):
    res2.append((0.9054*X1[i]+0.0012*X6[i]-0.2408-Y[i])**2)
print(sum(res2))
print(sum(res1))
print(sum(res2)/sum(res1))

