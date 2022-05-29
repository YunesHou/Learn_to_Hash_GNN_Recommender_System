import torch

class BaseLoss:
    def __init__(self):
        pass

    def get_loss(self, pred, label):
        raise NotImplementedError

    """ which should be negative first-order-deriv 
        divided by second-order-deriv """
    def get_gradient_boosted_target(self, prev_pred, label):
        raise NotImplementedError

class MSELoss(BaseLoss):
    def get_loss(self, pred, label):
        return torch.mean((pred-label)**2)

    def get_gradient_boosted_target(self, prev_pred, label):
        return label - prev_pred
