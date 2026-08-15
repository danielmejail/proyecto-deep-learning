import torch.nn as nn

class DNN(nn.Module):

    def __init__(self, input_size, hidden_layers, dropout):

        super().__init__()

        layers = []

        prev = input_size

        for neurons in hidden_layers:

            layers.append(nn.Linear(prev, neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev = neurons

        layers.append(nn.Linear(prev,1))

        self.network = nn.Sequential(*layers)

    def forward(self,x):

        return self.network(x)

