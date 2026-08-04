import torch
from torch import Tensor


class NormMeanStd:
    def __init__(self, shape: tuple[int,...], device: str = "cpu", epsilon: float = 1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
    
    def update(self, x: Tensor):
        if x.dim() == 1: x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, correction=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        self.mean += delta * batch_count / new_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + torch.square(delta) * self.count * batch_count / new_count
        self.var = m2 / new_count
        self.count =  new_count

    @torch.compile
    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class NormMinMax:
    def __init__(self, low: Tensor, high: Tensor, device: str = 'cpu'):
        self.low = low.to(device)
        self.high = high.to(device)
        self.scale = 1.0 / (self.high - self.low + 1e-8)

    @torch.compile
    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.low) * self.scale
