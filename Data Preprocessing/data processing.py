

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
text = pd.read_csv('Data.csv')
X = text.iloc[:,:-1].values
y = text.iloc[:,-1].values

# Fill up the missing data 
from sklearn.preprocessing import Imputer 
inputer = Imputer(missing_values="NaN",strategy ="mean",axis = 0)
inputer = inputer.fit(X[:,1:3])
X[:,1:3]=inputer.transform(X[:,1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
label_encoder = LabelEncoder()
X[:,0] = label_encoder.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features =[0])
X = onehotencoder.fit_transform(X).toarray()
y = label_encoder.fit_transform(y)

# Split dataset into the training set and testset 
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0 )

# Feature scaling 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)