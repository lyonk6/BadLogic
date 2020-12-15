# Here we play with numpy a little bit :)
import numpy as np

# Create some basic numpy arrays:
a=np.array([[1,2]])
b=np.array([[3],[4]])



# perform some matrix multiplication:
# a = [[1 2]]   b = [[3]
#                   [4]]
#
#print("Here are 'ab' and 'ba' respectfully: ")
#print(np.matmul(a,b))
#print(np.matmul(b,a))


# Convert our numpy object into a torch.
import torch

aTorch = torch.from_numpy(a.astype(np.float32)).requires_grad_()
bTorch = torch.from_numpy(b.astype(np.float32)).requires_grad_()
print("aTorch: ", aTorch)
print("bTorch: ", bTorch)

abTorch=aTorch.mm(bTorch)
baTorch=bTorch.mm(aTorch)
# Multiply two torch tensors together:

# Check out the Grad function: grad_fn=<MmBackward>)
print("\n")
print("A x B = [[3] x [[1 2]]:\n         [4]]")
print(baTorch.data)
print(baTorch.grad_fn)
print("\n")
print("A x B = [[1 2]] x [[3]:\n                   [4]]")
print(abTorch.data)
print(abTorch.grad_fn)
