import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader

import copy

from implementacion_de_la_arquitectura import DNN
from hiperparametros_optimizados import hiperparametros

# Importamos los datos
datadir = "../data/"
train_data_name = "data.csv"
train_target_name = "target.csv"

X_train = pd.read_csv(datadir + train_data_name, index_col=0)
y_train = pd.read_csv(datadir + train_target_name, index_col=0)

X_train_full_tensor = torch.tensor(np.array(X_train), \
        dtype=torch.float32)
y_train_full_tensor = torch.tensor(y_train.values, \
        dtype=torch.float32).reshape(-1, 1)
train_dataset_full = TensorDataset(X_train_full_tensor, \
        y_train_full_tensor)

train_loader_full = DataLoader(train_dataset_full, \
        batch_size=hiperparametros["batch_size"], shuffle=True)

modelo_final = DNN(
    input_size=X_train_full_tensor.shape[1],
    hidden_layers=hiperparametros["hidden_layers"],
    dropout=hiperparametros["dropout"]
)

criterion = torch.nn.MSELoss()
optimizer_final = torch.optim.Adam(
    modelo_final.parameters(),
    lr=hiperparametros["learning_rate"],
    weight_decay=hiperparametros["weight_decay"]
)


# Entrenamos con todo el dataset,
# guardando el checkpoint de la época con mejor loss
best_train_loss = float("inf")
best_state_dict = None

for epoch in range(100):
    modelo_final.train()
    running_loss = 0

    for X_batch, y_batch in train_loader_full:
        optimizer_final.zero_grad()
        outputs = modelo_final(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer_final.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader_full)

    if epoch_loss < best_train_loss:
        best_train_loss = epoch_loss
        best_state_dict = copy.deepcopy(modelo_final.state_dict())

# Guardamos el estado del modelo
torch.save(best_state_dict, "estado_de_pesos.pth")

