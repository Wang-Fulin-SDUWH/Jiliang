import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# 读取数据
# 第一问
data=pd.read_excel('Chapter2.xlsx')
data=data.values
X0=[]
Y=[]
for i in range(len(data)):
    X0.append(data[i][1])
    Y.append(data[i][0])
std_x=np.std(np.array(X0))
mean_x=np.mean(np.array(X0))
X=sm.add_constant(X0)
model=sm.OLS(Y,X)
results=model.fit()
# 回归系数标准差
print('回归模型拟合结果：')
print(results.summary())

# 第二问
pred=results.params[0]+results.params[1]*52000
print('浙江省2017年一般收入的点预测计算结果：',pred)
## 计算标准误 S.E.
s_e=0
for i in range(len(data)):
    s_e+=(results.fittedvalues[i]-Y[i])**2
std_err=(s_e/(len(data)-2))**0.5
print('标准误：',std_err)
# 计算得到标准误后，可以得出区间估计结果：(查表得知t0.025(37)约等于2.021)
delta1=std_x**2*(len(data)-1)
delta2=(52000-mean_x)**2
print(delta1,delta2)
delta_mean=2.021*std_err*sqrt(1/len(data)+delta1/delta2)
delta_ind=2.021*std_err*sqrt(1+1/len(data)+delta1/delta2)
print('浙江省2017年一般收入的区间预测计算结果：')
print('平均值置信度为95%的预测区间为：['+str(round(pred-delta_mean,3))+','+str(round(pred+delta_mean,3))+']')
print('个别值置信度为95%的预测区间为：['+str(round(pred-delta_ind,3))+','+str(round(pred+delta_ind,3))+']')
print('置信区间示意图：')
xxx,lower,upper=wls_prediction_std(results)
plt.plot(X0,lower,"r--")
plt.plot(X0,upper,"r--")
plt.plot(X0,results.fittedvalues,'g')
plt.scatter(X0,Y)
plt.show()

# 第三问
