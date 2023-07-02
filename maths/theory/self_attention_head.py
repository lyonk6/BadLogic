import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

block_size = 8
n_embd = 32

class Head(nn.Module):
    """ One head self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        # pytorch doesn't have a tril module, so we have to register our tril layer:
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        # compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        v = self.value(x)
        out = wei @ v
        return out