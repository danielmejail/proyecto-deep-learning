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

# Reservamos, del conjunto de train, un subconjunto de validación

# X_train_final, X_train_val, y_train_final, y_train_val = \
        # train_test_split( X_train_scaled, y_train_scaled, \
                            # test_size=.2, random_state=42 )

# Exportamos los datos modificados

train_data_name = "train_data.csv"
train_target_name = "train_target.csv"
train_data_scaled_name = "train_data_scaled.csv"
train_target_scaled_name = "train_target_scaled.csv"
test_data_name = "test_data.csv"
test_target_name = "test_target.csv"
test_data_scaled_name = "test_data_scaled.csv"
test_target_scaled_name = "test_target_scaled.csv"
# train_data_final_name = "train_data_final.csv"
# train_target_final_name = "train_target_final.csv"
# train_data_val_name = "train_data_val.csv"
# train_target_val_name = "train_target_val.csv"

features = X_train.columns
output = y_train.columns

# X_train.to_csv(datadir + train_data_name)
# y_train.to_csv(datadir + train_target_name)

# X_test.to_csv(datadir + test_data_name)
# y_test.to_csv(datadir + test_target_name)

X_train_scaled = pd.DataFrame(X_train_scaled)
y_train_scaled = pd.DataFrame(y_train_scaled)
X_train_scaled.columns = features
y_train_scaled.columns = output
X_train_scaled.to_csv(datadir + train_data_scaled_name)
y_train_scaled.to_csv(datadir + train_target_scaled_name)

X_test_scaled = pd.DataFrame(X_test_scaled)
y_test_scaled = pd.DataFrame(y_test_scaled)
X_test_scaled.columns = features
y_test_scaled.columns = output
X_test_scaled.to_csv(datadir + test_data_scaled_name)
y_test_scaled.to_csv(datadir + test_target_scaled_name)

# pd.DataFrame(X_train_final).to_csv(datadir + train_data_final_name)
# pd.DataFrame(y_train_final).to_csv(datadir + train_target_final_name)

# pd.DataFrame(X_train_val).to_csv(datadir + train_data_val_name)
# pd.DataFrame(y_train_val).to_csv(datadir + train_target_val_name)

# file_dict = {}
# file_dict["data"] = datafilename
# file_dict["target"] = targetfilename
# file_dict["train_data"] = train_data_name
# file_dict["train_target"] = train_target_name
# file_dict["train_data_scaled"] = train_data_scaled_name
# file_dict["train_target_scaled"] = train_target_scaled_name
# file_dict["test_data"] = test_data_name
# file_dict["test_target"] = test_target_name
# file_dict["test_data_scaled"] = test_data_scaled_name
# file_dict["test_target_scaled"] = test_target_scaled_name
# file_dict["train_data_final"] = train_data_final_name
# file_dict["train_target_final"] = train_target_final_name
# file_dict["train_data_val"] = train_data_val_name
# file_dict["train_target_val"] = train_target_val_name

