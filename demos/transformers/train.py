### First import the Tiny Shakespeare data
with open ('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("\nRead the tiny shakespeare source file, check the object type, (", type(text), "),")
print("then check the length of what should be a string: (", len(text), ").\n")

### Tokenizing and Splitting Data
# Use python's "set" and "list" constructors to tokenize the characters:
chars = sorted(list(set(text)))
vocab_size=len(chars)
print(''.join(chars))
print(vocab_size)

#define string-to-integer and integer-to-string functions:
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there"))  #[46, 47, 47, 1, 58, 46, 43, 56, 43]
print(decode(encode("hii there")))

# Create a tensor representing the entire Tiny Shakespeare dataset:
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# create training and testing datasets using a 90% split. We are 
# being ambitious here and trying to see if our network can make
# text identical to Shakespeare it's never seen before.
n = int(0.9*len(data))
train_data = data[:n]
val_data   = data[n:]
""" Pick up at 16 minutes """

## Building/Understanding the Data Loader
# select a block size of 8 characters:
block_size=8
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target  = y[t]
    

# The Bigram Language Model

# Attention

# Transformation