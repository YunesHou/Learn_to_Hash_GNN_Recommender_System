import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MySign

class BaseFunction(nn.Module):
    def __init__(self):
        super(BaseFunction, self).__init__()

    def forward(self, x):
        output = self.feature_extract(x)
        output = self.last_linear(output)
        return output

    def feature_extract(self, input):
        raise NotImplementedError

    def last_linear(self, input):
        raise NotImplementedError

    def force_limit(self):
        pass

class MLPHash(BaseFunction):
    def __init__(self, in_dim, hidden_dims, out_dim, use_bn):
        super(MLPHash, self).__init__()
        self.dims = [in_dim] + hidden_dims + [out_dim]
        self.extract_layers = nn.ModuleList([])
        for i in range(len(self.dims)-2):
            if use_bn == True:
                self.extract_layers.append(nn.BatchNorm1d(self.dims[i], affine=True))
            self.extract_layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            self.extract_layers.append(nn.LeakyReLU())

        self.last_linear_layer = nn.Linear(self.dims[-2], out_dim)
        self.temperature = nn.Parameter(0.5*torch.ones(1))
        self.sign = MySign()

    def feature_extract(self, input):
        output = input
        for layer in self.extract_layers:
            output = layer(output)
        return output

    def last_linear(self, input):
        output = self.last_linear_layer(input)
        output = torch.tanh(output * self.temperature)
        return output

    def binarize(self, input):
        output = self.sign.apply(input)
        return output

    def force_limit(self):
        with torch.no_grad():
            self.temperature.copy_(torch.clamp(self.temperature, min = 0.1))

class MLPFunc(BaseFunction):
    def __init__(self, in_dim, hidden_dims, out_dim, use_bn):
        super(MLPFunc, self).__init__()
        self.dims = [in_dim] + hidden_dims + [out_dim]
        self.extract_layers = nn.ModuleList([])
        for i in range(len(self.dims)-2):
            if use_bn == True:
                self.extract_layers.append(nn.BatchNorm1d(self.dims[i], affine=True))
            self.extract_layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            self.extract_layers.append(nn.LeakyReLU())

        self.last_linear_layer = nn.Linear(self.dims[-2], out_dim)

    def feature_extract(self, input):
        output = input
        for layer in self.extract_layers:
            output = layer(output)
        return output

    def last_linear(self, input):
        output = self.last_linear_layer(input)
        return output


