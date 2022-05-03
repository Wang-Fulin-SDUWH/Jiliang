import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import log
from math import sqrt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# 读取数据
# 第三问
data=pd.read_excel('Chapter2.xlsx')
data=data.values
X0=[]
Y=[]
for i in range(len(data)):
    X0.append(log(data[i][1]))
    Y.append(log(data[i][0]))
X=sm.add_constant(X0)
model=sm.OLS(Y,X)
results=model.fit()
# 回归系数标准差
print('回归模型拟合结果：')
print(results.summary())


plt.scatter(X0,Y,s=5)
X0=np.array(X0)
Y2=1.0308*X0-2.2571
plt.plot(X0,Y2,c='r')
plt.title('Fit Result')
# plt.title('Data Profile')
plt.show()