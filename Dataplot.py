import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data=pd.read_excel('Chapter2.xlsx')
data=data.values
X=[]
Y=[]
for i in range(len(data)):
    X.append(data[i][1])
    Y.append(data[i][0])
plt.scatter(X,Y,s=3)
plt.title('Data Profile')
plt.show()