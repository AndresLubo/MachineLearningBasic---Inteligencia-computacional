import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

url = '../bank-full.csv'
data = pd.read_csv(url)

# Tratamiento de la data
"""
Column marital
values: 

married = 0
divorced = 1
single = 2
"""
data.marital.replace(['married', 'divorced', 'single'], [0, 1, 2], inplace=True)

"""
Column education
values:

primary = 0
secundary = 1
tertiary = 2
unknown = 3
"""

data.education.replace(['primary', 'secondary', 'tertiary', 'unknown'], [0, 1, 2, 3], inplace=True)

"""
Column Default
value: 

no = 0
yes = 1
"""
data.default.replace(['no', 'yes'], [0, 1], inplace=True)

"""
Column housing
value: 

no = 0
yes = 1
"""

data.housing.replace(['no', 'yes'], [0, 1], inplace=True)
data.loan.replace(['no', 'yes'], [0, 1], inplace=True)
data.poutcome.replace(['failure', 'other', 'success', 'unknown'], [0, 1, 2, 3], inplace=True)

data.drop(['job', 'balance', 'contact', 'day', 'month', 'duration', 'pdays', 'previous', 'campaign'], axis= 1, inplace=True)

data.age.replace(np.nan, 41, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)

data.dropna(axis=0,how='any', inplace=True)
data.y.replace(['no', 'yes'], [0, 1], inplace=True)


# partir la data en dos
data_train = data[:728]
data_test = data[728:]

x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y) # 0 NO 1 Si

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y) # 0 No 1 Si

# Regresión Logística

# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')



# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')
