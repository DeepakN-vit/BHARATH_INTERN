import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt 

dataframe=pd.read_csv("Iris Data.csv")
X=dataframe.iloc[:,:-1].values
y=dataframe.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder()
y= onehot_encoder.fit_transform(y_encoded.reshape(-1, 1)).toarray()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(X_train,y_train)

from sklearn.preprocessing import PolynomialFeatures
polynomial_reg=PolynomialFeatures(degree=4)#the polynomial equation is go until 4 eg.a1+a2x1+a3x1^2+a4x1^3+a5x1^4
X_poly=polynomial_reg.fit_transform(X)
linear_reg_2=LinearRegression()
linear_reg_2.fit(X_poly,y)

y_pred=linear_reg.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error
mse=mean_squared_error(y_test,y_pred)
print("mean_squared_error:",mse)
r2_score=r2_score(y_test,y_pred)
print(r2_score)

plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred)
plt.title("Iris Data Classification")
plt.xlabel("Set of Characyters")
plt.ylabel("Species")
plt.show()