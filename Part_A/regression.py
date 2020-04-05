import matplotlib.pyplot as plt 
import numpy as np
import torch

x_values=np.array( [[2.7], [2.1], [3.9], [4.7],
                    [5.3], [5.9], [6.0], [9.1],
                    [8.0], [0.6], [3.3], [8.2]],
                   dtype=np.float32)                   

y_values=np.array( [[1.7], [1.4], [3.3], [4.1],
                    [5.2], [6.8], [4.0], [8.8],
                    [6.0], [0.5], [4.3], [8.1]],
                   dtype=np.float32)



plt.figure(figsize=(12, 8))
plt.scatter(x_values, y_values, label='Original data', s=250, c='g')
plt.legend()
# plt.show()

input_size =1
hidden_size=1
output_size=1

learning_rate = 1e-6

w1 = torch.rand(input_size,
                hidden_size,
                requires_grad=True)

print(w1.shape)

