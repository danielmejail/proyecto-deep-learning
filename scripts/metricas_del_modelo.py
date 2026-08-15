from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

import copy

from implementacion_de_la_arquitectura import DNN
from hiperparametros_optimizados import hiperparametros

# Importamos los datos
wdir = Path(__file__).resolve().parent
datadir = wdir.parent.joinpath("data")
train_data_file = datadir.joinpath("train_data_scaled.csv")
train_target_file = datadir.joinpath("train_target.csv")
test_data_file = datadir.joinpath("test_data_scaled.csv")
test_target_file = datadir.joinpath("test_target.csv")

X_train = pd.read_csv(train_data_file, index_col=0)
y_train = pd.read_csv(train_target_file, index_col=0)
X_test = pd.read_csv(test_data_file, index_col=0)
y_test = pd.read_csv(test_target_file, index_col=0)

X_train_full_tensor = torch.tensor(np.array(X_train), \
        dtype=torch.float32)
y_train_full_tensor = torch.tensor(y_train.values, \
        dtype=torch.float32).reshape(-1, 1)
train_dataset_full = TensorDataset(X_train_full_tensor, \
        y_train_full_tensor)
X_test_tensor = torch.tensor(np.array(X_test), \
        dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, \
        dtype=torch.float32).reshape(-1, 1)

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


# Entrenamos con todo el conjunto de train,
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

modelo_final.load_state_dict(best_state_dict)

### Métricas
metdir = wdir.parent.joinpath("metricas")
mse_file = metdir.joinpath("mse.csv")
predicciones_file = metdir.joinpath("predicciones.png")

### *mean squared error* estimado del modelo
modelo_final.eval()
with torch.no_grad():
    y_pred_test = modelo_final(X_test_tensor)
    mse_test = criterion(y_pred_test, y_test_tensor).item()

with open(mse_file, "w") as file:
    file.write(",MSE\n")
    file.write(f"0,{mse_test:.4f}")

### Comparación de predicciones en test con valores reales
plt.scatter(y_test_tensor, y_pred_test, s=8, label='DNN')
 
min_val = min(y_test_tensor.min(), y_pred_test.min())
max_val = max(y_test_tensor.max(), y_pred_test.max())
plt.plot([min_val, 5], [min_val, 5], color='red', linestyle='--', \
        label='Predicción perfecta')
plt.title('Valores reales vs. predichos')
plt.xlabel('Valor real')
plt.ylabel('Valor predicho')
plt.savefig(predicciones_file)

