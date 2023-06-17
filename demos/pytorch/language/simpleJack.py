import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class SimpleJack(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Return a batch, time, channel tensor (B, T, C).  Where our batch
        size is 4, our time window is 8 and our channel is 'vocab_size'.

        Note: "Negative Log Likelihood" is also called "Cross Entropy"
        https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81
        """
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Get the predictions, grab the last time step, apply softmax, 
        sample the distribution, then append that sample to a running
        sequence. 
        """
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # -> (B, C)
            probs = F.softmax(logits, dim=1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx