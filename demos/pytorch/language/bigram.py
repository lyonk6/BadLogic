import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

### Hyperparameters (parameters not changed by the model):
batch_size = 32
block_size = 8
max_iters = 2400
eval_iterval=300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("cuda?", device)
eval_iters = 200
n_embd = 32

### Import the Tiny Shakespeare data
with open ('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

### Tokenize using python's "set" and "list" constructors:
chars = sorted(list(set(text)))
vocab_size=len(chars)

### define string-to-integer and integer-to-string functions:
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

### Create training and validation tensors:
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data   = data[n:]

# This function grabs a random chunk from either a training
# set or a validation set.
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[1+i:1+i+block_size] for i in ix])
    return x, y

# model.eval() evaulates the performance of a model does not track the gradient. 
# model.train() DOES save the gradient and updates parameters accordingly. 
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear model head

    def forward(self, idx, targets=None):
        """
        Return a batch, time, channel tensor (B, T, C).  Where our batch
        size is 4, our time window is 8 and our channel is 'vocab_size'.

        Note: "Negative Log Likelihood" is also called "Cross Entropy"
        https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size) 


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Get the predictions, grab the last time step, apply softmax, 
        sample the distribution, then append that sample to a running
        sequence. 
        """
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # -> (B, C)
            probs = F.softmax(logits, dim=1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Create our "model" and assign it it the correct hardware:
model = BigramLanguageModel()
m = model.to(device)

# Pass model parameters to the optimizer so it knows what to update:
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iterval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val los {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss:
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
#print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


# TODO pick up at 1 hour