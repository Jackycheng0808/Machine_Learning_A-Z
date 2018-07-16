# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
text = pd.read_csv('Salary_Data.csv')
X = text.iloc[:,:-1].values
y = text.iloc[:,-1].values

# Split dataset into the training set and testset 
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3,random_state = 0 )

# Feature scaling (simple linear regression will take charge of normalization)
"""from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_X .fit_transform(y_train)
y_test = sc_X.transform(y_test) """ 

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_predict = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train)
plt.plot(X_train, regressor.predict(X_train),'red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test)
plt.plot(X_test, regressor.predict(X_test),'red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()