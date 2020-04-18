import matplotlib.pyplot as plt
import numpy as np
import torch
import tools

""" In this app, we train a very simple neural network to modify a given input 
    by a factor of phi. Use the ratio of neighbor Fibonacci numbers to create 
    increasing aproximations of phi as training data. Then plot what we find!"""

""" Start by creating our training data. Create corresponding torch vectors for
    this imput data: """

x = []
y = []
f0 = 0
fn = 1
N = 10
for i in range(1, N+1):
    temp = fn
    fn = f0 + fn
    f0 = temp
    x.append(i)
    y.append(i* fn/f0)

# Reshape x and y before making tensors [1, 2, ...] -> [[1], [2], ...]]
xA = np.asarray(x, dtype=np.float32).reshape(1, N)
yA = np.asarray(y, dtype=np.float32).reshape(1, N)
xT = torch.from_numpy(np.transpose(xA))
yT = torch.from_numpy(np.transpose(yA))

################################################################################    

""" Now let's define our very simple network of just a single node. Also we will
    define our learning rate. 

    Note the datatypes of w1, w2, y_pred is <class 'torch.Tensor'>    
    """

learning_rate = 1e-5   
w1 = torch.zeros(1, 1, requires_grad=True)
w2 = torch.zeros(1, 1, requires_grad=True)

y_pred = ""
predictions = []


for i in range(1, 2000):

    # perform the forward pass
    y_pred = xT.mm(w1).mm(w2)
    
    if i == 1:
        p001 = y_pred.detach().numpy()
        
    if i == 500:
        p500 = y_pred.detach().numpy()

    if i == 600:
        p600 = y_pred.detach().numpy()

    # calculate the error
    loss = (y_pred - yT).pow(2).sum()

    # Perform the backward pass
    loss.backward()

    """ Here we actually adjust the network. Note that we turn off autograd. 
        """
    
    with torch.no_grad():
        w1 -= learning_rate * w1.grad + learning_rate
        w2 -= learning_rate * w2.grad + learning_rate
        w1.grad.zero_()
        w2.grad.zero_()

final_pass = xT.mm(w1).mm(w2)
v = final_pass.detach().numpy()
################################################################################    
""" Note that xT and yT do no change. xT is always the input and yT is the 
    expected output.
    """
""" Next plot the data as a scatterplot and add our regression line too. """  
plt.figure(figsize=(12, 8))
plt.scatter(x, y, label='Original data', s=250, c='g')    
plt.plot(x, v, label = 'Fitted line')
plt.plot(x, p001, label = 'p001')
plt.plot(x, p500, label = 'p500')
plt.plot(x, p600, label = 'p600')
plt.legend()  
plt.show()    
