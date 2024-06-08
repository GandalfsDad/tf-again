import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    '''Single Head of Self Attention'''

    def __init__(self, embed_size, head_size, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.embed_size = embed_size
        self.block_size = block_size

        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()

        # C = head_size
        q = self.query(x) # (B, T, C)
        k = self.key(x) # (B, T, C)

        #Compute Attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=1) # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out

class MultiHeadAttention(nn.Module):
    '''Multi Head Self Attention'''

    def __init__(self, num_heads, embed_size, head_size, block_size, dropout) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_size, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    ''' feed forward component'''

    def __init__(self, N_EMBED, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    '''Transformer Block'''

    def __init__(self, num_heads, embed_size, block_size, dropout):
        super().__init__()
        head_size = embed_size // num_heads
        self.attention = MultiHeadAttention(num_heads, embed_size, head_size, block_size, dropout)
        self.feed_forward = FeedForward(embed_size, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x