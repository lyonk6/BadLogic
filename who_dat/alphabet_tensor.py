import torch
import numpy as np

# Now that we have the training data all set up in prep_data,
# here we will create a model to represent the alphabet with
# "one-hot-vectors"

# Note that while we only have 26 letters in the alphabet this
# input dataset contains capital and lowercase letters as well
# as th <space>, semicolon; comma and apostrophe. " .,;'" giving
# us 57 characters in total.

# one_by_n = np.zeros([26])
# print("One by N: ", one_by_n)
# 
# for i in range(0, 26):
#     temp_one_by_n = np.zeros([1, 26])
#     temp_one_by_n[0, i] = 1
#     print(chr(i+97), ": ",  temp_one_by_n)
#     #print(chr(i+97))
# 
# Turn a string of characters into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def nameToTensor(name):    
    name = name.strip()
    #TODO validate the characters are cocered to lowercase:
    t = torch.zeros([len(name), 26], dtype=torch.int32)
    i=0
    for c in name:
        if c.isalpha():
            c = c.lower()
            t[i][ord(c)-97]=1

        print(c, " ", type(i))
        i=i+1
    return t



# Print statements for testing :)
"""
print(nameToTensor('jones'))
print(nameToTensor('   Jones'))
print(nameToTensor(''.join([chr(96), chr(97), chr(98), chr(121), chr(122), chr(123)]))) #"""
