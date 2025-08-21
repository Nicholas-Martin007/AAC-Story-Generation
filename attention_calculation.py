import math

import torch
import torch.nn.functional as F

sentence = 'Saya mau menciptakan sebuah kisah sosial'

# Vocabulary
vocab = list(set(sentence.split()))
word2idx = {word: i for i, word in enumerate(vocab)}

# Convert sentence to indices
indices = [word2idx[word] for word in sentence.split()]
indices = torch.tensor(indices)  # [seq_len]

embedding_dim = 5
embedded_layer = torch.nn.Embedding(
    num_embeddings=len(vocab), embedding_dim=embedding_dim
)
embedded = embedded_layer(indices)  # [seq_len, d]

# QKV dimensions
d_q, d_k, d_v = 10, 10, 5
d = embedding_dim

# Learnable weights
W_query = torch.nn.Parameter(torch.rand(d_q, d))
W_key = torch.nn.Parameter(torch.rand(d_k, d))
W_value = torch.nn.Parameter(torch.rand(d_v, d))

# Compute Q, K, V with correct matmul
Q = embedded @ W_query.T  # [seq_len, d_q]
K = embedded @ W_key.T  # [seq_len, d_k]
V = embedded @ W_value.T  # [seq_len, d_v]

# Scaled dot-product attention
scores = (Q @ K.T) / math.sqrt(d_k)  # [seq_len, seq_len]
weights = F.softmax(scores, dim=-1)
attention_output = weights @ V  # [seq_len, d_v]

print('Attention output shape:', attention_output.shape)
print(attention_output)
