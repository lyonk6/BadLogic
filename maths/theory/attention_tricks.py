import torch
import torch.nn as nn
torch.manual_seed(42)

B, T, C = 4, 8, 2 # Batch, Time, Channel
x = torch.rand(B, T, C)
print(x)
print(x[1,:3])

# NB: 'bow' stands for "bag-of-words" which is a common term to
# describe a greedy amalgamation of words.
xbow = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev, 0)

print(xbow)
"""
According to Andrej Karpathy the key 'trick' in attention is the
weighted matrix multiplication which is made possible by with a 
function which applies zeroes to the upper right of a matrix:

# 1 4 7      # 1 0 0
# 2 5 8  ->  # 2 5 0
# 3 6 9      # 3 6 9
# 
"""
a = torch.tril(torch.ones(3,3))


"""
When "keepdim" is set to "False", the sum function squeezes the 
resulting matrix so the output tensor has 1 (or len(dim)) fewer 
dimension(s):

keepdim=True
tensor([[1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.3333, 0.3333, 0.3333]])

keepdim=False
tensor([[1.0000, 0.0000, 0.0000],
        [1.0000, 0.5000, 0.0000],
        [1.0000, 0.5000, 0.3333]])
"""

a = a/torch.sum(a,1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print("a=", a)
print("a=", b)
print("a @ b", c)

# Pick up at 52 mintes.