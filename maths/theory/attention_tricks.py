import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)

####################################################################
### Version 1. Simple attention trick.
####################################################################

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
weighted matrix multiplication used with a function which applies
zeroes to the upper right of a matrix:

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


# Note: this trilling trick is not necessary for self-attention. It
# is always applied to the decoder since the decoder is predicting
# the future, but the encoder usually doesn't need it.  

a = a/torch.sum(a,1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print("a=", a)
print("a=", b)
print("a @ b", c)


####################################################################
### Version 2. Full size matrices.
####################################################################
# wei is a (T, T) matrix and x is a (B, T, C) matrix...... So what's
# happening here!?!?  You can't multiply a "T,T" matrix by a "B,T,C"
# matrix!  As it turns out, pytorch is friendly enough to update our
# tensor and infer what dimension should be added so  we get:
#   "B x T x T" @  "B x T x C" from "T x T" @  "B x T x C".
# Which of course gives us a "B x T x C" matrix.

wei = torch.tril(torch.ones(T, T))
wei = wei/wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B,T,T) @ (B,T,C) -> (B,T,C)
truth = torch.allclose(xbow, xbow2) # should be true.
print("Is it true? ", truth)



####################################################################
### Version 3. Masked, tril(-inf), Softmax my dude.
####################################################################
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
truth = torch.allclose(xbow, xbow3) # should be true.



####################################################################
### Version 4. Self attention of a single head.
####################################################################
torch.manual_seed(1337)
B, T, C = 4, 8, 32 # Batch, Time, Channel
x = torch.randn(B,T,C)

"""
  The whole point of self attention is to make relevent information
  available where it is needed. The Batch dimensions however never
  talk to eachother.
"""

# single head self attention
head_size = 16
key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size)
k = key(x)
q = query(x)
wei = q @ k.transpose(-2,-1) # (B,T,16) @ (B,16,T) --> (B,T,T)

tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
v=value(x)
out = wei @ v
out = wei @ x
print(wei)
print(out.shape)

# What is attention anyway? 
# How about self-attention or cross-attention? 
# The example above is self-attention because each of k, q and v
# come from the same source, that is, these nodes are attending 
# to eachother and not a separate network. 

# Pick up at 1:19:00 