import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        """
        Return a batch, time, channel tensor (B, T, C).  Where our batch
        size is 4, our time window is 8 and our channel is 'vocab_size'.
        """
        logits = self.token_embedding_table = nn.Embedding(idx, 8)
        B, T, C = logits.shape
        

        # Note: "Negative Log Likelihood" is also called "Cross Entropy"
        # https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81
        loss   = F.cross_entropy(logits, targets)
        return logits, loss
