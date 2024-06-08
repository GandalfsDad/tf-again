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