from util import load_dataset, get_encode_decode, train_val_split
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import Block

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

#Parmaeters
BLOCK_SIZE = 16
BATCH_SIZE = 16
SEED = 42
MAX_TRAIN_STEPS = 2000
N_EMBED = 48
N_HEADS = 6
EVAL_INTERVAL = 10
LR = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_EVAL_STEPS = 200
N_BLOCKS = 4
DROPOUT = 0.2

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
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)
        
        self.blocks = nn.Sequential(*[Block(N_HEADS, N_EMBED, BLOCK_SIZE, DROPOUT) for _ in range(N_BLOCKS)])

        self.ln = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE)

    def forward(self, idx, targets = None):

        B, T = idx.shape

        token_emb = self.token_embedding(idx) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb # (B, T, C) pos emb is broadcasted

        x = self.blocks(x) # (B, T, C)
        #x = self.ff(x) # (B, T, C)

        logits = self.lm_head(x) # (B, T, VOCAB_SIZE)

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
            # Trim to last BLOCK_SIZE characters
            idx_cond = idx[:, -BLOCK_SIZE:] # (B, BLOCK_SIZE)
            # get predictions
            logits, loss = self(idx_cond)
            # get last time step
            logits = logits[:,-1, :] # (B, C)
            # get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample next character
            idx_next = torch.multinomial(probs, 1) # (B, 1)
            # append to idx
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)   

@torch.no_grad()
def estimate_loss():
    out = {'train': 0, 'val': 0}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(MAX_EVAL_STEPS)
        for k in range(MAX_EVAL_STEPS):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    
    model.train()
    return out

for step in range(MAX_TRAIN_STEPS):

    #Eval if needed
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step: {step}, Train Loss: {losses['train']} Val Loss: {losses['val']}")

    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

print(decode(model.generate(torch.zeros((1,1), dtype=torch.long, device=device), 1000)[0].tolist()))
