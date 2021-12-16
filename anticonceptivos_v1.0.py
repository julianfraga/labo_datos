# -*- coding: utf-8 -*-
"""
Created on Thu May  6 18:58:42 2021

@author: Julián
"""

import matplotlib.pylab as plt
import numpy as np 
from sklearn.linear_model import LogisticRegression 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score
import os
from sklearn.feature_selection import SelectKBest
#%%
# Todo lo de siempre: cargar df, armar features etc
os.chdir(r'D:\Documentos\UBA\LaboDatos2021\anticoncepcion indonesia')
filename= 'anticoncepcion_indonesia.csv'
df = pd.read_csv(filename)
df.dropna(inplace=True)
feats=list(df.columns)[0:-1] # Preparo mi lista de features

# =============================================================================
# one-hot encoding para la categoría de ocupaciones
# =============================================================================
indice=np.logical_not(df['ocupacion_hombre']==' ')
df_filtrado=df[indice] # Tiro las entradas vacías si las hubiera
encoder = OneHotEncoder(sparse=False) #Armo el objeto encoder
ocupacion = np.array(df_filtrado['ocupacion_hombre']).reshape(-1, 1) # aca pasamos a un vector la serie de pandas, como es usual
encoder.fit(ocupacion) # fiteo
print(encoder.categories_) # estas son las columnas binarias del nuevo encoding
ocupacion_hot = encoder.transform(ocupacion) # obtenemos la mariz binaria
print(ocupacion_hot)

# Armo la lista de distintas ocupaciones y un DF
ocupaciones=list(['ocupacion_hombre_1','ocupacion_hombre_2', 'ocupacion_hombre_3', 'ocupacion_hombre_4'])
ocupacion=pd.DataFrame(ocupacion_hot,columns=ocupaciones)
ocupacion.astype('int')

feats.remove('ocupacion_hombre') # Saco la columna categórica
feats.extend(ocupaciones)       # Adjunto los nuevos cuatro features
for i in range(len(ocupaciones)):
    df[ocupaciones[i]]=ocupacion_hot[:,i]


for i in range(len(ocupaciones)): # Chequeo no haber perdido datos en el medio
    if (df[ocupaciones[i]].value_counts())[1]==list((df['ocupacion_hombre'].value_counts(sort=False)))[i]:
        print('Todo bien')
    else:
        print('Hiciste lío')

# =============================================================================
# Armo matrices X e Y con features elevados de 1 a 'n'
# =============================================================================

df['usa_anticonceptivos']=0 
indice =  df['metodo_anticonceptivo'] > 1  # esto me da los valores del indice para los cuales hay precipitacion mayor a 0
df.loc[indice, 'usa_anticonceptivos'] = 1 # entonces para esos valores del indice pongo 1, porque en el dia correspondiente, llovio

x=df[feats].values
y=df['usa_anticonceptivos'].values

# Cargo la data de la simulación para la determinación de hiperparámetros
n_values = np.arange(3,6,1)
c_values = np.arange(4,6.2, 0.2)
mean_BA_train=np.loadtxt('mean_n3a6_c4a6_train.csv', delimiter=',')
mean_BA_test=np.loadtxt('mean_n3a6_c4a6_test.csv', delimiter=',')
#Encuentro los mejores hiper parámetros
n_max,c_max= n_values[np.where(mean_BA_test==mean_BA_test.max())[0]],c_values[np.where(mean_BA_test==mean_BA_test.max())[1]]
n_max, c_max= float(n_max), float(c_max)
print('Los mejores valores de n y c para el clasificador con penalización tipo L1 encontrados son n={}, c={:.2f}'.format(n_max, c_max))

# Armo mi matriz X con los features elevados de 1 a 'n'
x_concatenado=x
for i in np.arange(2,n_max+1):
    x_concatenado = np.concatenate((x_concatenado,x**i), axis=1)
    
# =============================================================================
# Selecciono las mejores K features de mi dataframe
# =============================================================================
Kbest = 30 # los mejores K que voy a retener

skf = StratifiedKFold(n_splits=5, shuffle=True) # 5 folds es un número típico si tenemos suficientes datos. Pedimos shuffle=True para que sea al azar la separación en subgrupos
skf.get_n_splits(X, y) # arma los folds a partir de los datos

auc_values_fs =  []  # aca es donde van a ir a parar los indices de los features seleccionados en cada fold
selected_features= np.array([]).reshape(0,X.shape[1]) # aca es donde van a ir a parar los AUCs de cada fold. El reshape es para poder concatenar luego.

targets = np.array([])   
indices = np.array([]) 
scores = np.array([]) 
for train_index, test_index in skf.split(X, y): # va generando los indices que corresponden a train y test en cada fold
    X_train, X_test = X[train_index], X[test_index] # arma que es dato de entrenamiento y qué es dato de evaluación
    y_train, y_test = y[train_index], y[test_index]     # idem con los targets

    scaler = MinMaxScaler() # escaleo por separado ambos sets
    scaler.fit(X_train) 
    X_train = scaler.transform(X_train)

    scaler = MinMaxScaler() # escaleo por separado ambos sets
    scaler.fit(X_test) 
    X_test = scaler.transform(X_test)

    selector = SelectKBest(k=Kbest) # por defecto, usa el F score de ANOVA y los Kbest features
    selector.fit(X_train, y_train) # encuentro los F scores 
    X_train_fs = selector.transform(X_train) # me quedo con los features mejor rankeados en el set de entrenamiento
    X_test_fs = selector.transform(X_test) # me quedo con los features mejor rankeados en el set de evaluacion
    features = np.array(selector.get_support()).reshape((1,-1)) # esto me pone True si la variable correspondiente fue seleccionada y False sino

    selected_features =  np.concatenate((selected_features,features),axis=0)

    regLog = LogisticRegression(penalty = 'l1',C=c_max, max_iter=10000) # Inicializamos nuevamente el modelo. max_iter es la cantidad de iteraciones maximas del algoritmo de optimizacion de parametros antes de detenerse.
    regLog.fit(X_train_fs, y_train) # Ajustamos el modelo con los datos de entrenamiento


    probas_test = regLog.predict_proba(X_test_fs)  # probabilidades con datos de evaluación
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, probas_test[:,1]) # para plotear curva ROC con datos de entrenamiento
    auc_test = roc_auc_score(y_test, probas_test[:,1]) #  AUC con datos de evaluación
    auc_values_fs.append(auc_test)
    indices = np.concatenate((indices,test_index),axis=0)
    targets = np.concatenate((targets,y_test),axis=0)
    scores = np.concatenate((scores,probas_test[:,1]),axis=0)
print('El AUC promedio usando los Kbest feats es:',np.mean(auc_values_fs))
feats_producto_temp=['edad_mujer', 'educacion_mujer', 'educacion_hombre', 'numero_hijos', 'religion_mujer', 'mujer_trabaja', 'nivel_vida', 'exposicion_medios', 'ocupacion_hombre_1', 'ocupacion_hombre_2', 'ocupacion_hombre_3', 'ocupacion_hombre_4', 'edad_mujer_cuadrado', 'educacion_mujer_cuadrado', 'educacion_hombre_cuadrado', 'numero_hijos_cuadrado', 'religion_mujer_cuadrado', 'mujer_trabaja_cuadrado', 'nivel_vida_cuadrado', 'exposicion_medios_cuadrado', 'ocupacion_hombre_1_cuadrado', 'ocupacion_hombre_2_cuadrado', 'ocupacion_hombre_3_cuadrado', 'ocupacion_hombre_4_cuadrado', 'edad_mujer_cubo', 'educacion_mujer_cubo', 'educacion_hombre_cubo', 'numero_hijos_cubo', 'religion_mujer_cubo', 'mujer_trabaja_cubo', 'nivel_vida_cubo', 'exposicion_medios_cubo', 'ocupacion_hombre_1_cubo', 'ocupacion_hombre_2_cubo', 'ocupacion_hombre_3_cubo', 'ocupacion_hombre_4_cubo']
feats_producto=[]
for i in range(len(features[0])):
    if features[0][i]==True:
        feats_producto.append(feats_producto_temp[i])
# print('los {} mejores features elegidos son:'.format(Kbest), feats_producto)

plt.bar(np.arange(0,X.shape[1]),np.sum(selected_features,axis=0))
plt.title('Seleccion de features')
plt.xlabel('Feature')
plt.ylabel('Folds')

from numpy.random import shuffle # para shufflear el vector
y_shuffled = y.copy() # creo una copia del vector de targets, porque shuffle lo pisa
shuffle(y_shuffled) # shuffleo

skf = StratifiedKFold(n_splits=5, shuffle=True) # 5 folds es un número típico si tenemos suficientes datos. Pedimos shuffle=True para que sea al azar la separación en subgrupos
skf.get_n_splits(X, y_shuffled) # arma los folds a partir de los datos

auc_values = [] # aca es donde van a ir a parar los AUCs de cada fold
scores_shuffled = np.array([])     # aca es donde van a ir a parar los scores computados para todos los casos
indices_shuffled = np.array([])    # aca es donde van a ir a parar los indices correspondientes a las entradas de scores
targets_shuffled = np.array([])    # aca es donde van a ir a parar los targets en el orden de la validacion cruzada

for train_index, test_index in skf.split(X, y_shuffled): # va generando los indices que corresponden a train y test en cada fold
    X_train, X_test = X[train_index], X[test_index] # arma que es dato de entrenamiento y qué es dato de evaluación
    y_train, y_test = y_shuffled[train_index], y_shuffled[test_index]     # idem con los targets

    regLog_1 = LogisticRegression(penalty = 'l1', C=c_max, max_iter=10000) # Inicializamos nuevamente el modelo. max_iter es la cantidad de iteraciones maximas del algoritmo de optimizacion de parametros antes de detenerse.
    regLog_1.fit(X_train, y_train) # Ajustamos el modelo con los datos de entrenamiento

    probas_test = regLog_1.predict_proba(X_test)  # probabilidades con datos de evaluación
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, probas_test[:,1]) # para plotear curva ROC con datos de entrenamiento
    auc_test = roc_auc_score(y_test, probas_test[:,1]) #  AUC con datos de evaluación

    auc_values.append(auc_test)
    scores_shuffled  = np.concatenate((scores_shuffled ,probas_test[:,1]),axis=0)
    indices_shuffled  = np.concatenate((indices_shuffled ,test_index),axis=0)
    targets_shuffled  = np.concatenate((targets_shuffled ,y_test),axis=0)



# print("Estos son los valores AUC para cada fold en el MODELO NULO:")
# print(auc_values)
print("Estos es el promedio de todos los AUC en el MODELO NULO:",np.mean(auc_values))

fpr, tpr, thresholds = roc_curve(targets, scores)
fpr_shuffled, tpr_shuffled, thresholds_shuffled = roc_curve(targets_shuffled, scores_shuffled)

fig, ax = plt.subplots(figsize = (10,7))
ax.set_title('Verdaderos positivos vs. falsos positivos')
ax.plot(fpr,tpr, label='Modelo real')
ax.plot(fpr_shuffled,tpr_shuffled, label='Modelo nulo')

ax.set_xlabel('Tasa de falsos positivos') # Etiqueta del eje x
ax.set_ylabel('Tasa de verdaderos positivos') # Etiqueta del eje y
plt.legend()
#%%
# =============================================================================
# EL CLASIFICADOR
# =============================================================================
