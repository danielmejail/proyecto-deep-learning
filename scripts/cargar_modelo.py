import numpy as np
import torch

from implementacion_de_la_arquitectura import DNN
from hiperparametros_optimizados import hiperparametros

N_FEATURES = 8
pesosfile = "estado_de_pesos.pth"

# Probamos cargar el estado del modelo entrenado y utilizarlo para predicción
modelo = DNN( input_size=N_FEATURES, \
        hidden_layers=hiperparametros["hidden_layers"], \
        dropout=hiperparametros["dropout"] )

modelo.load_state_dict(torch.load(pesosfile))
modelo.eval()

x_scaled = torch.tensor(np.array([12., 2., .1, .1, 5., .3, -1, -2]), \
        dtype=torch.float32)
y_pred = modelo(x_scaled).detach().numpy()[0]

print(y_pred)

