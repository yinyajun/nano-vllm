import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile  # 小而频繁、图稳定、维度固定的模块
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [x * W_gate, x * W_up] 只有x * W_gate部分需要activation
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
