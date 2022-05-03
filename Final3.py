import pandas as pd
import numpy as np
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def cRun(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    print("使用math库：r：", SSR / SST, "r-squared：", (SSR / SST) ** 2)
    return (SSR / SST) ** 2


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

X_pd=pd.DataFrame({'X1':X1,'X2':X2,'X3':X3,'X4':X4,'X5':X5,'X6':X6})
print('解释变量之间的相关系数：\n',X_pd.corr())


X11=np.column_stack((X_intercept,X1))
model1=sm.OLS(Y,X11)
results1=model1.fit()
print(results1.summary())
print(cRun(X1,Y))

X22=np.column_stack((X_intercept,X2))
model2=sm.OLS(Y,X22)
results2=model2.fit()
print(results2.summary())
print(cRun(X2,Y))

X33=np.column_stack((X_intercept,X3))
model3=sm.OLS(Y,X33)
results3=model3.fit()
print(results3.summary())
print(cRun(X3,Y))

X44=np.column_stack((X_intercept,X4))
model4=sm.OLS(Y,X44)
results4=model4.fit()
print(results4.summary())
print(cRun(X4,Y))

X55=np.column_stack((X_intercept,X5))
model5=sm.OLS(Y,X55)
results5=model5.fit()
print(results5.summary())
print(cRun(X5,Y))

X66=np.column_stack((X_intercept,X6))
model6=sm.OLS(Y,X66)
results6=model6.fit()
print(results6.summary())
print(cRun(X6,Y))

