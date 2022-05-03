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
plt.scatter(X,Y,s=5)
X=np.array(X)
Y2=0.1918*X-227.0518
plt.plot(X,Y2,c='r')
plt.title('Fit Result')
plt.show()