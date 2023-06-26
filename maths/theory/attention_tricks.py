import torch
import torch.nn as nn
torch.manual_seed(42)

B, T, C = 4, 8, 2 # Batch, Time, Channel
x = torch.rand(B, T, C)
print(x)
print(x[1,:3])

# NB: 'bow' stands for "bag-of-words" which is a common term
# to describe a greedy amalgamation of words.
xbow = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev, 0)

print(xbow)

a = torch.tril(torch.ones(3,3))
# TODO What does keepdim do?
a = a/torch.sum(a,1, keepdim=True)
print(a)