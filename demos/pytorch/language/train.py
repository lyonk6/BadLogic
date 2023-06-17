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
#print(data[:1000])

# create training and testing datasets using a 90% split. We are 
# being ambitious here and trying to see if our network can make
# text identical to Shakespeare it's never seen before.
n = int(0.9*len(data))
train_data = data[:n]
val_data   = data[n:]
""" Pick up at 16 minutes """

### Building the Data Loader
# Select a block size of 8 characters. Ensure that both 'x' and
# 'y' are of length 8. Since this is a prediction model the 
# target data (y) should be one character ahead of the input.
# 
block_size=8
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target  = y[t]
    print(f"when input is {context} the target: {target}")

# Create a "batch dimension" whose only purpose is to aid us in
# processing this data in parallel. 
torch.manual_seed(1337)
batch_size = 4 # sequences to be processed in parallel.
block_size = 8 # How big are the input/output vectors?

# This function grabs a random chunk from either a training
# set or a validation set.
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[1+i:1+i+block_size] for i in ix])
    return x, y

xb, yb = get_batch('train')
print("inputs:  ", xb.shape)
print("outputs: ", yb.shape)
print(xb)
print(yb)

print('----')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target  = yb[b,t]
        print(f"When the input is {context.tolist()} the target: {target}")

### The Bigram Language Model
import simpleJack
model = simpleJack.SimpleJack(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))

# Instantiate an optimizer:
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32
for steps in range(10000):
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss:
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))


### Attention
# 

### Transformation
# 
