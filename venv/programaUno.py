import numpy as np
import matplotlib.pyplot as pit
import pandas as pd

dataset = pd.read_csv('venv/Data.csv')

x = dataset.iloc[:,:-1].values #feature matrix
y = dataset.iloc[:,-1].values #Variable dependiente

print(x)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(x[:,1:3]) # Identificar y transformar celdas vacias
x[:,1:3] = imputer.transform(x[:,1:3]) # Actualiza la feature matrix con la media

print(x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=1)

print('------------ Test & Train Sets --------------')
print('------------ Train X --------------')
print(x_train)
print('------------ Test X --------------')
print(x_test)
print('------------ Train Y --------------')
print(y_train)
print('------------ Test Y --------------')
print(y_test)

#Comentario prueba

