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
    X1.append(np.log(data[i][2]))
    X2.append(data[i][3])
    X3.append(np.log(data[i][4]))
    X4.append(data[i][5])
    X5.append(data[i][6])
    X6.append(data[i][7])
    Y.append(np.log(data[i][1]))
X_intercept=np.ones(len(data))
X1=np.array(X1)
X2=np.array(X2)
X3=np.array(X3)
X4=np.array(X4)
X5=np.array(X5)
X6=np.array(X6)
X=np.column_stack((X_intercept,X1,X2,X3,X4,X5,X6)) # 添加截距
Y=np.array(Y)


model = LinearRegression()
X12=np.column_stack((X_intercept,X1,X2))
model.fit(X12,Y)
model12=sm.OLS(Y,X12)
results6=model12.fit()
print(results6.summary())
print(model.score(X12,Y))

model = LinearRegression()
X13=np.column_stack((X_intercept,X1,X3))
model.fit(X13,Y)
model13=sm.OLS(Y,X13)
results6=model13.fit()
print(results6.summary())
print(model.score(X13,Y))

model = LinearRegression()
X14=np.column_stack((X_intercept,X1,X4))
model.fit(X14,Y)
model14=sm.OLS(Y,X14)
results6=model14.fit()
print(results6.summary())
print(model.score(X14,Y))

model = LinearRegression()
X15=np.column_stack((X_intercept,X1,X5))
model.fit(X15,Y)
model15=sm.OLS(Y,X15)
results6=model15.fit()
print(results6.summary())
print(model.score(X15,Y))

model = LinearRegression()
X16=np.column_stack((X_intercept,X1,X6))
model.fit(X16,Y)
model16=sm.OLS(Y,X16)
results6=model16.fit()
print(results6.summary())
print(model.score(X16,Y))

#Round 2
print('Stop here'*100)

model = LinearRegression()
X142=np.column_stack((X_intercept,X1,X6,X2))
model.fit(X142,Y)
model142=sm.OLS(Y,X142)
results6=model142.fit()
print(results6.summary())
print(model.score(X142,Y))

model = LinearRegression()
X143=np.column_stack((X_intercept,X1,X6,X3))
model.fit(X143,Y)
model143=sm.OLS(Y,X143)
results6=model143.fit()
print(results6.summary())
print(model.score(X143,Y))

model = LinearRegression()
X145=np.column_stack((X_intercept,X1,X6,X4))
model.fit(X145,Y)
model145=sm.OLS(Y,X145)
results6=model145.fit()
print(results6.summary())
print(model.score(X145,Y))

model = LinearRegression()
X146=np.column_stack((X_intercept,X1,X6,X5))
model.fit(X146,Y)
model146=sm.OLS(Y,X146)
results6=model146.fit()
print(results6.summary())
print(model.score(X146,Y))
