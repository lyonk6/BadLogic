import numpy as np

a = [1, 2, 3, 4]
A = np.asarray(a, dtype=np.float32)
AT = np.transpose(A)

b = [[0, 1], [2, 3]]
B = np.asarray(b, dtype=np.float32)
BT = np.transpose(B)

print("Here is B: \n", B)
print("")
print("Here is B transposed: \n", BT)

print("")
print("")
print("")

print("Here is A: \n", A)
print("")
print("Here is A transposed: \n", AT)

k = np.arange(9)
K = k.reshape(1,9)
KT = np.transpose(K)

print("")
print("Here is k:  \n", k)
print("Can we reshape an array? ", k.shape)


print("")
print("Here is K:  \n", K)
print("Can we reshape an array? ", K.shape)


print("")
print("Here is KT: \n", KT)
print("Can we reshape an array? ", KT.shape)
