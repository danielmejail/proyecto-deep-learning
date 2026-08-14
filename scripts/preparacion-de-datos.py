import os

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importamos los datos

datadir = "../data/"
datafilename = "data.csv"
targetfilename = "target.csv"

try:
    data = pd.read_csv(datadir + datafilename, index_col=0)
    target = pd.read_csv(datadir + targetfilename, index_col=0)
except IOError:
    (data, target) = fetch_california_housing(return_X_y=True, as_frame=True)
    data.to_csv(datadir + datafilename)
    target.to_csv(datadir + targetfilename)

# Separamos los datos entre train y test

X_train, X_test, y_train, y_test = \
        train_test_split( data, target, test_size=.2, random_state=42 )

# Normalizamos los datos

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(pd.DataFrame(y_train))
y_test_scaled = scaler_y.transform(pd.DataFrame(y_test))


