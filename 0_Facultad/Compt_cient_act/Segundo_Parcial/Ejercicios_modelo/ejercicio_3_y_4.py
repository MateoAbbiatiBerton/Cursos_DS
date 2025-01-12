import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from ISLP.models import (ModelSpec as MS, summarize)
import statsmodels.api as sm

eleccion=int(input(f'Elegir el modelo a ejecutar \n Para regresion lineal ingrese:\t\t 1 \n Para regresion logistica ingresar:\t 2 \nEleccion:\t'))

if eleccion == 1:
    df = pd.read_csv('./Salary_data.csv')
    y = df['Salary']
    x = df[['Experience Years']]
    lr = LinearRegression()
    lr.fit(x,y)

    a = input('Años de experiencia:')

    b = lr.predict(pd.DataFrame({'Experience Years': [a]}))

    print(f'La prediccion del salario es: {round(b.item(), 2)}')


elif eleccion == 2:
    df = pd.read_csv('./breast_cancer.csv')
    y = df['Class']
    x = df.drop(['Class'], axis=1)
    positive = y == 4
    diseño = MS(x)
    X = diseño.fit_transform(x)
    Y = positive
    glm = sm.GLM(Y,X,
                family=sm.families.Binomial())
    resultados = glm.fit()
    
    l = ['Clump Thickness',
    'Uniformity of Cell Size',
    'Uniformity of Cell Shape',
    'Marginal Adhesion',
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses']
    resp = [int(input(f'{i}: ')) for i in l]
    
    g=pd.DataFrame(data = {i:[j] for i,j in zip(l,resp)})
    g = sm.add_constant(g, has_constant='add')
    print(f'La predicion es: {int(resultados.predict(g))}')

else: print('Opcion no valida')
