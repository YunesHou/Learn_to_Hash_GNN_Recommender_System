from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
from config import CFG
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np

def dot(X1, X2):
    return X1 @ X2.T


def init_logger(log_file=CFG.save_dir + 'train.log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

class VectorDataset(Dataset):
    def __init__(self, W, transform = None):
        self.W = W
        self.transform = transform
        
    def __len__(self):
        return self.W.shape[0]
    
    def __getitem__(self, idx):
        w = self.W[idx]
        if self.transform:
            return self.transform(w)
        else:
            return w
        
class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()
    
    @staticmethod
    def backward(ctx, grad):
        return grad

def BinaryRegularization(code):
    return torch.mean(torch.abs(torch.abs(code)-1))

def VarianceRegularization(code):
    cov = torch.triu(code.T.detach()@code, diagonal=1)/code.shape[0]
    return torch.mean(torch.abs(cov))
 
def write_down(prefix, state_dict):
  layers = list(state_dict.keys())
  layers.sort()
  for layer in layers:
    path = os.path.join(prefix, layer+".data")
    weights = state_dict[layer].cpu().detach().numpy()
    if len(weights.shape) == 0:
      continue
    weights = weights.astype(np.double)
    weights.astype('double').tofile(path)


