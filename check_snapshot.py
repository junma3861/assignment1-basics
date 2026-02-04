import pickle

with open('tests/_snapshots/test_train_bpe_special_tokens.pkl', 'rb') as f:
    data = pickle.load(f)
    
print('Reference vocab_values sample (first 20):')
for i, val in enumerate(sorted(data['vocab_values'])[:20]):
    print(f'{i}: {repr(val)}')

print()
vocab_size = len(data['vocab_values'])
print(f'Total vocab size: {vocab_size}')
print()
crlf_present = b'\r\n' in data['vocab_values']
special_present = b'<|endoftext|>' in data['vocab_values']
print(f'Has CRLF in vocab: {crlf_present}')
print(f'Has special token: {special_present}')
print()

# Check what's in our current vocab
from cs336_basics.train_bpe import train_bpe
vocab, merges = train_bpe(
    'tests/fixtures/tinystories_sample_5M.txt',
    vocab_size=1000,
    special_tokens=['<|endoftext|>'],
)

print('\nOur vocab_values sample (first 20):')
for i, val in enumerate(sorted(vocab.values())[:20]):
    print(f'{i}: {repr(val)}')

print()
our_vocab_size = len(vocab.values())
print(f'Our total vocab size: {our_vocab_size}')
print()
our_crlf = b'\r\n' in vocab.values()
our_special = b'<|endoftext|>' in vocab.values()
print(f'Our vocab has CRLF: {our_crlf}')
print(f'Our vocab has special token: {our_special}')

# Show what extra items we have
actual_values = set(vocab.values())
expected_values = data['vocab_values']
extra = actual_values - expected_values
print(f'\nExtra items in our vocab ({len(extra)} total):')
for item in sorted(extra)[:20]:
    print(f'  {repr(item)}')
