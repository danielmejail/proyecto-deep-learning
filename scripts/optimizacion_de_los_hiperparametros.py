import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

from itertools import product

from implementacion_de_la_arquitectura import DNN

# Importamos los datos
datadir = "../data/"
train_data_name = "train_data_scaled.csv"
train_target_name = "train_target.csv"
# test_data_name = "test_data_scaled.csv"
# test_target_name = "test_target.csv"

X_train = pd.read_csv(datadir + train_data_name, index_col=0)
y_train = pd.read_csv(datadir + train_target_name, index_col=0)

# X_test = pd.read_csv(datadir + test_data_name, index_col=0)
# y_test = pd.read_csv(datadir + test_target_name, index_col=0)

# Reservamos, del conjunto de train, un subconjunto de validación
# para realizar la optimización de los hiperparámetros del modelo
X_train_final, X_train_val, y_train_final, y_train_val = \
        train_test_split( X_train, y_train, test_size=.2, random_state=42 )

# Convertimos a tensores
X_final_tensor = torch.tensor(np.array(X_train_final), \
        dtype=torch.float32)
y_final_tensor = torch.tensor(y_train_final.values, \
        dtype=torch.float32).reshape(-1, 1)

X_val_tensor = torch.tensor(np.array(X_train_val), \
        dtype=torch.float32)
y_val_tensor = torch.tensor(y_train_val.values, \
        dtype=torch.float32).reshape(-1, 1)

# X_test_tensor = torch.tensor(X_test, \
        # dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test.values, \
        # dtype=torch.float32).reshape(-1, 1)

# Datasets
final_dataset = TensorDataset(X_final_tensor, y_final_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)


### Función de entrenamiento para optimización de hiperparámetros

def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                epochs=100):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss/len(train_loader)
        train_losses.append(train_loss)

        model.eval()

        running_val = 0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                pred = model(X_batch)

                loss = criterion(pred,y_batch)

                running_val += loss.item()

        val_loss = running_val/len(val_loader)

        val_losses.append(val_loss)

    return train_losses, val_losses


### Mella para optimización de hiperparámetros

param_grid = {

    "hidden_layers":[
        [64],
        [128],
        [64, 32],
        [128, 64],
        [128, 64, 32],
        [32, 128, 64]
    ],

    "learning_rate":[.001, .004, .007, .01],

    "batch_size":[32, 64],

    "dropout":[.0, .1, .2],

    "weight_decay": [.0001]

}


### Optimización de los hiperparámetros

best_mse = float("inf")
best_model = None
best_params = None

for hidden, lr, batch, dropout, wd in product(
        param_grid["hidden_layers"],
        param_grid["learning_rate"],
        param_grid["batch_size"],
        param_grid["dropout"],
        param_grid["weight_decay"]):

    train_loader = DataLoader(
        final_dataset,
        batch_size=batch,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        shuffle=False
    )

    model = DNN(
        input_size=X_final_tensor.shape[1],
        hidden_layers=hidden,
        dropout=dropout
    )

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=100
    )

    mse = min(val_losses)

    if mse < best_mse:

        best_mse = mse

        best_model = model

        best_train_losses = train_losses
        best_val_losses = val_losses

        best_params = {

            "hidden_layers":hidden,
            "learning_rate":lr,
            "batch_size":batch,
            "dropout":dropout,
            "weight_decay":wd

        }

### Reporte de los mejores hiperparámetros
with open("hiperparametros_optimizados.py", "w") as file:
    file.write("hiperparametros = ")
    file.write(str(best_params))

### Curva de train contra validación
train_loader = DataLoader(train_dataset, \
        batch_size=best_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, \
        batch_size=best_params["batch_size"], shuffle=False)

model_check = DNN(input_size=X_final_tensor.shape[1], \
        hidden_layers=best_params["hidden_layers"], \
        dropout=best_params["dropout"])
criterion = nn.MSELoss()
optimizer_check = torch.optim.Adam(model_check.parameters(), \
        lr=best_params["learning_rate"], \
        weight_decay=best_params["weight_decay"])

train_losses_check, val_losses_check = \
        train_model( model_check, train_loader, val_loader, \
            criterion, optimizer_check, epochs=100 )

metdir = "../metricas/"
plt.figure(figsize=(8, 5))
plt.plot(train_losses_check, label='Pérdida en entreamiento (MSE)')
plt.plot(val_losses_check, label='Pérdida en test (MSE)')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.title('Curva de entrenamiento y evaluación del modelo')
plt.legend()
plt.tight_layout()
plt.savefig(metdir + "rendimiento.png")

