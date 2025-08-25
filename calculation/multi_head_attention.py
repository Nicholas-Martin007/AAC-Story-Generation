import torch
import torch.nn.functional as F

h = 4
d_model = 8
d_k, d_v = d_model // h, d_model // h

sentence = 'Saya mau menciptakan sebuah kisah sosial yang baru'

# Vocabulary
vocab = list(set(sentence.split()))
word2idx = {word: i for i, word in enumerate(vocab)}

# Convert sentence to indices
indices = [word2idx[word] for word in sentence.split()]
indices = torch.tensor(indices)  # [sentence_length]

sentence_length = len(indices)

# Embedding layer
embedded_layer = torch.nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=d_model,
)

input_embedding = embedded_layer(indices)  # [seq_len, d_model]

w_q = torch.nn.Linear(d_model, d_model)
w_k = torch.nn.Linear(d_model, d_model)
w_v = torch.nn.Linear(d_model, d_model)

Q = w_q(input_embedding)  # [seq_len, d_model]
K = w_k(input_embedding)  # [seq_len, d_model]
V = w_v(input_embedding)  # [seq_len, d_model]

# [seq_len, d_model] -> [seq_len, h, d_k] -> [h, seq_len, d_k]
Q = Q.view(sentence_length, h, d_k).transpose(0, 1)
K = K.view(sentence_length, h, d_k).transpose(0, 1)
V = V.view(sentence_length, h, d_v).transpose(0, 1)

# Q @ K^T -> [h, seq_len, seq_len]
scores = Q @ K.transpose(-2, -1) / (d_k**0.5)
weights = F.softmax(scores, dim=-1)

attention_heads = weights @ V

# [seq_len, h, d_v] -> [seq_len, d_model]
attention_heads = (
    attention_heads.transpose(0, 1)
    .contiguous()
    .view(sentence_length, d_model)
)

print(f'input_embedding {input_embedding.detach()}')

print(f'w_q weight {w_q.weight}')
print(f'w_k weight {w_k.weight}')
print(f'w_v weight {w_v.weight}')


print(f'Q {Q.detach()}')
print(f'K {K.detach()}')
print(f'V {V.detach()}')

print('Multi-Head Attention output:')
print(attention_heads.detach())
