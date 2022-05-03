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

