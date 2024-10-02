import numpy as np
import pandas as pd

df=pd.read_csv('kc_house_data.csv')



df=df.drop('date',axis=1)

x=df['sqft_living'].values
X=x.reshape(-1,1)
y=df['price'].values
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

slreg=LinearRegression()
slreg.fit(X_train,y_train)

y_pred=slreg.predict(X_test)

import matplotlib.pyplot as plt

plt.scatter(X_train,y_train,color="green")
plt.plot(X_train,slreg.predict(X_train),color='yellow')
plt.title("Space vs Price Training")
plt.xlabel('Space')
plt.ylabel('Price')

plt.show()

plt.scatter(X_test,y_test,color="green")
plt.plot(X_test,y_pred,color='yellow')
plt.title("Space vs Price Testing")
plt.xlabel('Space')
plt.ylabel('Price')

plt.show()


