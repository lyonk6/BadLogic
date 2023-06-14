import numpy as np


# Calculating the scaled dot product is a 3 step process:
#   1. Compute the dot-product of Q and K.T
#   2. Divide by the square root of |K|
#   3. Apply a softmax function
#   4. Apply the weighted values function
def crapy_scaled_dot_product(Q, K, V):
    d = np.sqrt(K.size)
    p = np.dot(Q, K.T)
    return (p/d) * V


import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, keys, values):
    """
    Calculates scaled dot-product attention.
    
    Args:
        query (torch.Tensor): Query tensor of shape (query_length, d_model).
        keys (torch.Tensor): Key tensor of shape (key_length, d_model).
        values (torch.Tensor): Value tensor of shape (key_length, d_model).
        
    Returns:
        torch.Tensor: Scaled dot-product attention output tensor of shape (query_length, d_model).
    """
    d_model = query.size(-1)
    
    # Calculate scaled dot product
    scores = torch.matmul(query, keys.transpose(-2, -1)) / (d_model ** 0.5)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Weight the values by the attention weights
    output = torch.matmul(attention_weights, values)
    
    return output


# Example inputs
query_length = 4
key_length = 5
d_model = 3

query = torch.randn(query_length, d_model)
keys = torch.randn(key_length, d_model)
values = torch.randn(key_length, d_model)
print("query: ", query)
print("keys:  ", keys)
print("values:", values)
# Calculate scaled dot-product attention
attention_output = scaled_dot_product_attention(query, keys, values)

# Print the attention output shape
print(attention_output.shape)