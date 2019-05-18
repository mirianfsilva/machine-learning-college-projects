
#%%
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn import svm
from sklearn import metrics
from sklearn import ensemble
from sklearn import neighbors
from sklearn import feature_selection


#%%
# Naive Bayes: Apenas um experimento para servir de baseline
# Decision Tree: Variar a altura máxima da árvore (incluindo permitir altura ilimitada) e 
# mostrar os resultados graficamente
# SVM: Avaliar os kernels linear, sigmoid, polinomial e RBF
# k-NN: Variar o número k de vizinhos e mostrar os resultados graficamente
# Random Forest: Variar o número de árvores e mostrar os resultados graficamente.
# Gradient Tree Boosting: Variar o número de iterações e mostrar os resultados graficamente. 


#%%
#Input filepath 
INPUT_FILEPATH = "koi_data.csv"
TARGET = "koi_disposition"
N_FEATURES = 40

df = pd.read_csv(INPUT_FILEPATH)
df = df.drop(["kepoi_name"], axis=1)

print("lines: {}".format(df.shape[0]))
print("rows: {}".format(df.shape[1]))
print("Missing data: {}".format(df.isnull().sum().sum()))


print("\n InputFile:")
with pd.option_context("max_columns", 40): # Limita o numero de cols mostradas
    display(df.head(20))

# list features
features = list(df.columns)
features.remove(TARGET)
print("Target: {}".format(TARGET))

# print("Features:")
# print("\n".join(["  " + x for x in features]))


