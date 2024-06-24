import torch
import torch.nn as nn

# architecture: 3 layer MLP: input -> 1024 -> 500 -> 50 -> output
# activation: ReLU
# regularization: dropout(0.2) on middle layer
# normalization: batch norm on each layer

class StockwishEvalMLP_Mod(nn.Module):

    def __init__(self, num_features, num_units_hidden, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_units_hidden = num_units_hidden

        # the architecture is 3 hidden layers, each with 2048 units
        lin = nn.Linear(num_features, num_units_hidden)
        self.hidden_1 = nn.Sequential(
            lin,
            nn.BatchNorm1d(num_units_hidden),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2),
        )
        lin = nn.Linear(num_units_hidden, 500)
        self.hidden_2 = nn.Sequential(
            lin,
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        lin = nn.Linear(500, 50)
        self.hidden_3 = nn.Sequential(
            lin,
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2),
        )

        # added sigmoid hopefully makes the training better
        self.linear_out = nn.Sequential(
            torch.nn.Linear(50, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        return self.linear_out(x)


# architecture: 3 layer MLP:  input -> 2048 -> 2048 -> 2048 -> output
# activation: ReLU
# regularization: dropout(0.2) on middle layer
# normalization: batch norm on each layer

class StockwishEvalMLP(nn.Module):

    def __init__(self, num_features, num_units_hidden, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_units_hidden = num_units_hidden

        # the architecture is 3 hidden layers, each with 2048 units
        lin = nn.Linear(num_features, num_units_hidden)
        self.hidden_1 = nn.Sequential(
            lin,
            nn.BatchNorm1d(num_units_hidden),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2),
        )
        lin = nn.Linear(num_units_hidden, num_units_hidden)
        self.hidden_2 = nn.Sequential(
            lin,
            nn.BatchNorm1d(num_units_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        lin = nn.Linear(num_units_hidden, num_units_hidden)
        self.hidden_3 = nn.Sequential(
            lin,
            nn.BatchNorm1d(num_units_hidden),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2),
        )

        # added sigmoid hopefully makes the training better
        self.linear_out = nn.Sequential(
            torch.nn.Linear(num_units_hidden, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        return self.linear_out(x)