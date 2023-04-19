import torch
from torch import Tensor

class TSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor, t: Tensor) -> Tensor:
        assert (t > 0.0).all()
        maxes = torch.max(input, dim=self.dim, keepdim=True).values
        input_minus_maxes = input - maxes

        w = torch.relu(input_minus_maxes + t)

        x_exp = w * torch.exp(input_minus_maxes)
        x_exp_sum = torch.sum(x_exp, dim=self.dim, keepdim=True)
        out = x_exp / x_exp_sum

        return out

class RSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.tsoftmax = TSoftmax(dim=dim)

    @classmethod
    def calculate_t(cls, input: Tensor, r: Tensor, dim: int = -1, eps: float = 1e-8):
        # r represents what is the fraction of zero values that we want to have
        assert ((0.0 <= r) & (r <= 1.0)).all()

        maxes = torch.max(input, dim=dim, keepdim=True).values
        input_minus_maxes = input - maxes

        zeros_mask = torch.exp(input_minus_maxes) == 0.0
        zeros_frac = zeros_mask.sum(dim=dim, keepdim=True).float() / input_minus_maxes.shape[dim]

        q = torch.clamp((r - zeros_frac) / (1 - zeros_frac), min=0.0, max=1.0)
        x_minus_maxes = input_minus_maxes * (~zeros_mask).float()
        t = -torch.quantile(x_minus_maxes, q.view(-1), dim=dim, keepdim=True).detach()
        t = t.squeeze(dim).diagonal(dim1=-2, dim2=-1).unsqueeze(-1) + eps

        return t

    def forward(self, input: Tensor, r: Tensor):
        t = RSoftmax.calculate_t(input, r, self.dim, self.eps)
        return self.tsoftmax(input, t)
