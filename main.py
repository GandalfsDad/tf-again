from util import load_dataset, get_encode_decode, train_val_split
import torch
import torch.nn as nn
from torch.nn import functional as F

data = load_dataset()


# Get the unique characters in the text
chars = sorted(list(set(data)))

# Create a mapping from unique characters to indices
VOCAB_SIZE = len(chars)
encode, decode = get_encode_decode(chars)

data = torch.tensor(encode(data), dtype=torch.long)

train_data, val_data = train_val_split(data, 0.9)

print(f'Vocab size: {VOCAB_SIZE}')
print(f'Train data size: {len(train_data)}')
print(f'Val data size: {len(val_data)}')

BLOCK_SIZE = 8
BATCH_SIZE = 32
SEED = 4

torch.manual_seed(SEED)

def get_batch(split):
    data = train_data if split == 'train' else val_data

    # Generate Offsets
    start_idx = torch.randint(0, len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([data[idx:idx+BLOCK_SIZE] for idx in start_idx])
    y = torch.stack([data[idx+1:idx+BLOCK_SIZE+1] for idx in start_idx])

    return x, y

# B Batch
# T Time
# C Channels

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        #idx, and targets are of shape (B, T)

        logits = self.embedding(idx) # (B, T, C)

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss 
    
    def generate(self, idx, length):
        #idx is of shape (B, T)
        #length is the number of characters to generate
        # model generates to B, T+length
        B, T = idx.shape

        for _ in range(length):
            logits, loss = self(idx)
            # get last time step
            logits = logits[:,-1, :] # (B, C)
            # get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample next character
            idx_next = torch.multinomial(probs, 1) # (B, 1)
            # append to idx
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(VOCAB_SIZE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)   

for step in range(1000):
    x, y = get_batch('train')
    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f'Step: {step}, Loss: {loss.item()}')