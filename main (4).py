import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")

X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
lin_regressor=LinearRegression()
lin_regressor.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_regressor=PolynomialFeatures(degree=4)
X_poly=poly_regressor.fit_transform(X)
lin_regressor2=LinearRegression()
lin_regressor2.fit(X_poly,y)


plt.scatter(X,y,color='red')
plt.plot(X,lin_regressor2.predict(X_poly))
plt.title("Truth Or Bluff(Polynomial Regression)")
plt.xlabel("Position")
plt.ylabel('Salery')
plt.show()
