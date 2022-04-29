import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

url = '../diabetes.csv'
data = pd.read_csv(url)

# Tratamiento de la data

data.Pregnancies.replace(np.nan, 4, inplace=True)
rangos_pregnancies = [0, 5, 10, 15, 20]
nombres_pregnancies = [1, 2,3, 4]
data.Pregnancies = pd.cut(data.Pregnancies, rangos_pregnancies, labels=nombres_pregnancies)

data.Glucose.replace(np.nan, 121, inplace=True)
rangos_glucose = [0, 50, 100, 150, 200]
nombres_glucose = [1, 2,3, 4]
data.Glucose = pd.cut(data.Glucose, rangos_glucose, labels=nombres_glucose)

data.BloodPressure.replace(np.nan, 69, inplace=True)
rangos_bloodPressure = [0, 50, 100, 150]
nombres_bloodPressure = [1, 2, 3]
data.BloodPressure = pd.cut(data.BloodPressure, rangos_bloodPressure, labels=nombres_bloodPressure)

data.SkinThickness.replace(np.nan, 21, inplace=True)
rangos_skinThickness = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
nombres_skinThickness = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]
data.SkinThickness = pd.cut(data.SkinThickness, rangos_skinThickness, labels=nombres_skinThickness)

data.Insulin.replace(np.nan, 78, inplace=True)
rangos_insulin = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
nombres_insulin = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]
data.Insulin = pd.cut(data.Insulin, rangos_insulin, labels=nombres_insulin)

data.BMI.replace(np.nan, 32, inplace=True)
rangos_bmi = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
nombres_bmi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]
data.BMI = pd.cut(data.BMI, rangos_bmi, labels=nombres_bmi)

data.drop(['DiabetesPedigreeFunction'], axis=1, inplace=True)

data.Age.replace(np.nan, 33, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.Age = pd.cut(data.Age, rangos, labels=nombres)


data.dropna(axis=0,how='any', inplace=True)



# partir la data en dos
data_train = data[:253]
data_test = data[253:]

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) # 0 NO 1 Si

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome) # 0 No 1 Si

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

# classifiers SGD

#seleccionar el modelo
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=7600)

#entrenar el modelo
clf.fit(x_train, y_train)

#Métricas
print('*'*50)
print('classifier, SGD')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {clf.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {clf.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {clf.score(x_test_out, y_test_out)}')

# MLPClassifier

#seleccionar el modelo
mpl = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

#entrenar el modelo
mpl.fit(x_train, y_train)

#Métricas
print('*'*50)
print('MLPClassifier')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {mpl.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {mpl.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {mpl.score(x_test_out, y_test_out)}')