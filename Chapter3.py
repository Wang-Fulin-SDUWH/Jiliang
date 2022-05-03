import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# 读取数据
# 第一问
data=pd.read_excel('Chapter3.xlsx')
data=data.values
X=[]
Y=[]
for i in range(len(data)):
    X.append(data[i][1:3])
    Y.append(data[i][0])
X_intercept=np.ones(len(data))
X=np.array(X)
X=np.column_stack((X_intercept,X)) # 添加截距
Y=np.array(Y)
model=sm.OLS(Y,X)
results=model.fit()
print(results.summary())