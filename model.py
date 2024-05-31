import torch
import torch.nn as nn

class StockwishEvalMLP(nn.Module):

    def __init__(self, num_features, num_units_hidden, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_units_hidden = num_units_hidden

        # the architecture is 3 hidden layers, each with 2048 units
        lin = nn.Linear(num_features, num_units_hidden)
        lin.weight.detach().normal_(0.0, 1.0)
        self.hidden_1 = nn.Sequential(
            lin,
            nn.BatchNorm1d(num_units_hidden),
            nn.ELU(inplace=True),
        )
        # initialize weights to normal with zero mean and unit variance
        #self.hidden_1.weight.detach().normal_(0.0, 1.0)
        #self.hidden_1.bias.detach().zero_()
        lin = nn.Linear(num_units_hidden, num_units_hidden)
        lin.weight.detach().normal_(0.0, 1.0)
        self.hidden_2 = nn.Sequential(
            lin,
            nn.BatchNorm1d(num_units_hidden),
            nn.ELU(inplace=True),
        )
        # initialize weights to normal with zero mean and unit variance
        #self.hidden_2.weight.detach().normal_(0.0, 1.0)
        #self.hidden_2.bias.detach().zero_()
        lin = nn.Linear(num_units_hidden, num_units_hidden)
        lin.weight.detach().normal_(0.0, 1.0)
        self.hidden_3 = nn.Sequential(
            lin,
            nn.BatchNorm1d(num_units_hidden),
            nn.ELU(inplace=True),
        )
        # initialize weights to normal with zero mean and unit variance
        #self.hidden_3.weight.detach().normal_(0.0, 1.0)
        #self.hidden_3.bias.detach().zero_()

        self.linear_out = torch.nn.Linear(num_units_hidden, num_classes)

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        return self.linear_out(x)

if __name__ == '__main__':
    x = torch.randn((2, 768))
    model = StockwishEvalMLP(num_features=768, num_units_hidden=100, num_classes=1)
    y = model(x)
    print(y)
    print(torch.cuda.is_available())