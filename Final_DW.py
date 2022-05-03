import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AR
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_white
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

data=pd.read_excel('Data.xlsx')
data=data.values
X1=[]
X6=[]
Y=[]
res=[]
res_sq=[]
rho1=1.3590
rho2=-0.3637
for i in range(len(data)-2):
    X1.append(np.log(data[i+2][2])-rho1*np.log(data[i+1][2])-rho2*np.log(data[i][2]))
    X6.append(data[i+2][7]-rho1*data[i+1][7]-rho2*data[i][7])
    Y.append(np.log(data[i+2][1])-rho1*np.log(data[i+1][1])-rho2*np.log(data[i][1]))
X_intercept=np.ones(len(data)-2)
X1=np.array(X1)
X6=np.array(X6)
X=np.column_stack((X_intercept,X1,X6)) # 添加截距
Y=np.array(Y)

model=sm.OLS(Y,X)
results=model.fit()
print(results.summary())

for i in range(len(Y)):
    res.append(0.7417*X1[i]+0.0012*X6[i]+0.0149-Y[i])

# LM检验--二阶
res1=res[1:]
res2=res[2:]
res.pop(-1)
res.pop(-1)
res1.pop(-1)
res=np.array(res)
res1=np.array(res1)
res2=np.array(res2)
Xlm=np.column_stack((X_intercept[:-2],X1[:-2],X6[:-2],res1,res2))
Ylm=res

model_lm=sm.OLS(Ylm,Xlm)
results=model_lm.fit()
print(results.summary())


# def ar_estimation(sample,p):
#     matrix_x=np.zeros((len(sample)-p,p))
#     array=sample.reshape(len(sample))
#     j=0
#     for i in range(0,len(sample)-p):
#         matrix_x[i,0:p]=array[j:j+p]
#         j+=1
#     matrix_y=np.array(array[p:len(sample)])
#     matrix_y=matrix_y.reshape(len(sample)-p,1)
#     #AR系数的表达式：A=(X^TX)^-1 X^T Y
#     coef=np.dot(np.dot(np.linalg.inv(np.dot(matrix_x.T,matrix_x)),matrix_x.T),matrix_y)
#     return coef
# coefs=ar_estimation(np.array(res),2)
# print(coefs)
