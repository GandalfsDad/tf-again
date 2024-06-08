from util import load_dataset, get_encode_decode, train_val_split
import torch

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
BATCH_SIZE = 4
SEED = 4

torch.manual_seed(SEED)

def get_batch(split):
    data = train_data if split == 'train' else val_data

    # Generate Offsets
    start_idx = torch.randint(0, len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([data[idx:idx+BLOCK_SIZE] for idx in start_idx])
    y = torch.stack([data[idx+1:idx+BLOCK_SIZE+1] for idx in start_idx])

    return x, y

