import pandas as pd
import numpy as np


# Create training data for a neural network to
A = np.zeros(pow(2,3), dtype=int)
B = np.zeros(pow(2,3), dtype=int)
is_AND = np.zeros(pow(2,3), dtype=int) # default is OR.
output = np.zeros(pow(2,3), dtype=int)


position = 0
for j in range(0, 2):
    for k in range(0, 2):
         for l in range(0, 2):
             A[position] = l
             B[position] = k
             is_AND[position] = j
             position += 1


for i in range(0, len(output)):
    if is_AND[i] == 0:
        if A[i] + B[i] >= 1:
            output[i]=1
        else:
            output[i]=0
    else:
        if A[i] + B[i] == 2:
            output[i]=1
        else:
            output[i]=0


training_data = pd.DataFrame({"A": A, "B": B, "is_AND": is_AND, "output":output})
#print(training_data)
