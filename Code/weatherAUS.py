import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

url = '../weatherAUS.csv'
data = pd.read_csv(url)


# Tratamiento de la data
data.MinTemp.replace(np.nan, 12, inplace=True)
rangos_minTemp = [-85, -35, 15, 65, 115, 165, 215, 265, 315, 365, 400]
nombres_minTemp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data.MinTemp = pd.cut(data.MinTemp, rangos_minTemp, labels=nombres_minTemp)


data.MaxTemp.replace(np.nan, 23, inplace=True)
rangos_maxTemp = [-48, 2, 52, 102, 152, 202, 252, 302, 352, 402, 452, 500]
nombres_maxTemp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
data.MaxTemp = pd.cut(data.MaxTemp, rangos_maxTemp, labels=nombres_maxTemp)

data.Rainfall.replace(np.nan, 2, inplace=True)
rangos_rainfall = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
nombres_rainfall = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data.Rainfall = pd.cut(data.Rainfall, rangos_rainfall, labels=nombres_rainfall)

data.Evaporation.replace(np.nan, 5, inplace=True)
rangos_evaporation = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
nombres_evaporation = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data.Evaporation = pd.cut(data.Evaporation, rangos_evaporation, labels=nombres_evaporation)

data.Sunshine.replace(np.nan, 8, inplace=True)
rangos_sunshine = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
nombres_sunshine = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data.Sunshine = pd.cut(data.Sunshine, rangos_sunshine, labels=nombres_sunshine)

data.WindGustSpeed.replace(np.nan, 40, inplace=True)
rangos_wgSpeed = [0, 20, 40, 60, 80, 100, 120, 140, 150]
nombres_wgSpeed = [1, 2, 3, 4, 5, 6, 7, 8]
data.WindGustSpeed = pd.cut(data.WindGustSpeed, rangos_wgSpeed, labels=nombres_wgSpeed)

data.WindSpeed9am.replace(np.nan, 14, inplace=True)
rangos_ws9am = [0, 20, 40, 60, 80, 100, 120, 130]
nombres_ws9am = [1, 2, 3, 4, 5, 6, 7]
data.WindSpeed9am = pd.cut(data.WindSpeed9am, rangos_ws9am, labels=nombres_ws9am)

data.WindSpeed3pm.replace(np.nan, 19, inplace=True)
rangos_ws3pm = [0, 20, 40, 60, 80, 87]
nombres_ws3pm = [1, 2, 3, 4, 5]
data.WindSpeed3pm = pd.cut(data.WindSpeed3pm, rangos_ws3pm, labels=nombres_ws3pm)

data.Humidity9am.replace(np.nan, 69, inplace=True)
rangos_h9am = [0, 20, 40, 60, 80, 100]
nombres_h9am = [1, 2, 3, 4, 5]
data.Humidity9am = pd.cut(data.Humidity9am, rangos_h9am, labels=nombres_h9am)

data.Humidity3pm.replace(np.nan, 51, inplace=True)
rangos_h3pm = [0, 20, 40, 60, 80, 100]
nombres_h3pm = [1, 2, 3, 4, 5]
data.Humidity3pm = pd.cut(data.Humidity3pm, rangos_h3pm, labels=nombres_h3pm)

data.Pressure9am.replace(np.nan, 1018, inplace=True)
rangos_p9am = [950, 970, 990, 1010, 1030, 1050]
nombres_p9am = [1, 2, 3, 4, 5]
data.Pressure9am = pd.cut(data.Pressure9am, rangos_p9am, labels=nombres_p9am)

data.Pressure3pm.replace(np.nan, 1015, inplace=True)
rangos_p3pm = [950, 970, 990, 1010, 1030, 1050]
nombres_p3pm = [1, 2, 3, 4, 5]
data.Pressure3pm = pd.cut(data.Pressure3pm, rangos_p3pm, labels=nombres_p3pm)

data.Cloud9am.replace(np.nan, 4, inplace=True)
rangos_c9am = [0, 2, 4, 6, 8, 10]
nombres_c9am = [1, 2, 3, 4, 5]
data.Cloud9am = pd.cut(data.Cloud9am, rangos_c9am, labels=nombres_c9am)

data.Cloud3pm.replace(np.nan, 4, inplace=True)
rangos_c3pm = [0, 2, 4, 6, 8, 10]
nombres_c3pm = [1, 2, 3, 4, 5]
data.Cloud3pm = pd.cut(data.Cloud3pm, rangos_c3pm, labels=nombres_c3pm)

data.Temp9am.replace(np.nan, 17, inplace=True)
rangos_t9am = [-7, 3, 13, 23, 33, 43]
nombres_t9am = [1, 2, 3, 4, 5]
data.Temp9am = pd.cut(data.Temp9am, rangos_t9am, labels=nombres_t9am)

data.Temp3pm.replace(np.nan, 22, inplace=True)
rangos_t3pm = [-5, 5, 15, 25, 35, 45, 50]
nombres_t3pm = [1, 2, 3, 4, 5, 6]
data.Temp3pm = pd.cut(data.Temp3pm, rangos_t3pm, labels=nombres_t3pm)

data.RainToday.replace(['No', 'Yes'], [0, 1], inplace=True)

data.RISK_MM.replace(np.nan, 2, inplace=True)
rangos_rmm = [0, 50, 100, 150, 200, 250, 300, 350, 400]
nombres_rmm = [1, 2, 3, 4, 5, 6, 7, 8]
data.RISK_MM = pd.cut(data.RISK_MM, rangos_rmm, labels=nombres_rmm)

data.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)

data.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', ], axis=1, inplace=True)

data.dropna(axis=0,how='any', inplace=True)

print(data)

# partir la data en dos
data_train = data[:19962]
data_test = data[19962:]

x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow) # 0 NO 1 Si

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow) # 0 No 1 Si

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


