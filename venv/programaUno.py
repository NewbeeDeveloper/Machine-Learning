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

#Comentario prueba

