import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_kdd import dot

class BaseDecoderFunction(nn.Module):
    def __init__(self):
        super(BaseDecoderFunction, self).__init__()

    """
    h_o and f_q should be n*d matrices 
    """
    def forward(self, h_o, f_q):
        raise NotImplementedError
    #    vector = self.get_bit_vector(h_o, f_q)
    #    return torch.sum(vector, dim = 1)

    #def get_bit_vector(self, h_o, f_q):
    #    raise NotImplementedError

    def force_limit(self):
        pass

    def load_partial_params(self, model, begin, end):
        pass
"""
class HammingDistanceDecoder(BaseDecoderFunction):
    def __init__(self, is_01):
        self.is_01 = is_01
        super(HammingDistanceDecoder, self).__init()

    def forward(self, h_o, f_q):
        if self.is_01 == True:
            return torch.abs(h_o-f_q)
        else:
            return torch.abs(h_o-f_q)/2
"""
class WeightedInnerProductDecoder(BaseDecoderFunction):
    def __init__(self, dim):
        super(WeightedInnerProductDecoder, self).__init__()
        self.dim = dim
        self.weights = nn.Parameter(torch.rand(dim))

    def forward(self, h_o, f_q):
        return dot(h_o * self.weights, f_q)

    def force_limit(self):
        with torch.no_grad():
            self.weights.copy_(torch.clamp(self.weights, min=0))

    def load_partial_params(self, model, begin, end):
        self.weights.data[begin:end] = model.weights.data[0:end-begin]

class LHTIPSDecoder(BaseDecoderFunction):
    def __init__(self, dim):
        super(LHTIPSDecoder, self).__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.rand(dim))
        self.beta = nn.Parameter(torch.rand(dim))

    def forward(self, h_o, f_q):
        rescaled_h_o = F.relu(h_o) * self.alpha - F.relu(-h_o) * self.beta
        return rescaled_h_o

    def force_limit(self):
        with torch.no_grad():
            self.alpha.copy_(torch.clamp(self.alpha,min=0))
            self.beta.copy_(torch.clamp(self.beta, min=0))

    def load_partial_params(self, model, begin, end):
        self.alpha.data[begin:end] = model.alpha.data[0:end-begin]
        self.beta.data[begin:end] = model.beta.data[0:end-begin]
