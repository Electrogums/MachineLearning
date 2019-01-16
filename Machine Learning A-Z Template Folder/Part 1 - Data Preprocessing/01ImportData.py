#Data Preprocesing 

#Import the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

#Importing the dataset 
dataset = pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

#Taking care of missing data 
#Tratamos los datos nullos rellenandolos
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding categorical data
#Le asignamos un numero a los paises del dataset para que no influya en los resultados.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

onehoteconder=OneHotEncoder(categorical_features=[0])
X=onehoteconder.fit_transform(X).toarray()

labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#Splitting, the dataset into the trainging set and test set
#dividimos el dataset en dos un 20% para test y dejamos un 80% para entrenar al modelo
from sklearn.model_selection import train_test_split
X_train,X_test,Y_traing,Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature Scaling 
#Escalando la informacion para que una variable que en este caso es el salario,
#No domine a la variable peque;a osea la edad.
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)











