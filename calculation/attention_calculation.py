import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 123
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# FFN
class FFNModel(nn.Module):
    def __init__(self, d_v):
        super(FFNModel, self).__init__()

        self.fc1 = nn.Linear(
            d_v, d_v * 4
        )  # dff = 2048, d_model = 512, 2048 / 512 = 4
        self.fc2 = nn.Linear(d_v * 4, d_v)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out


# def positional_encoding(seq_len, d_model):
#     pos = torch.arange(seq_len).unsqueeze(
#         1
#     )  # shape [seq_len, 1]
#     i = torch.arange(d_model).unsqueeze(0)  # shape [1, d_model]
#     angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
#     angles = pos * angle_rates

#     pe = torch.zeros(seq_len, d_model)
#     pe[:, 0::2] = torch.sin(
#         angles[:, 0::2]
#     )  # even indices → sine
#     pe[:, 1::2] = torch.cos(
#         angles[:, 1::2]
#     )  # odd indices → cosine
#     return pe


# Khusus bagian encoder
# pos_enc = positional_encoding(sentence_length, d_model)
# input_embedding = input_embedding + pos_enc


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

embedded_layer = torch.nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=d_model,
)

input_embedding = embedded_layer(
    indices,
)  # [sentence_length, d_model]

# menggunakan Q = X * W_T + b ##https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
w_q = torch.nn.Linear(d_model, d_k)  # [d_k, d_model]
w_k = torch.nn.Linear(d_model, d_k)  # [d_k, d_model]
w_v = torch.nn.Linear(d_model, d_v)  # [d_v, d_model]

Q = w_q(input_embedding)  # [seq_len, d_q]
K = w_k(input_embedding)  # [seq_len, d_k]
V = w_v(input_embedding)  # [seq_len, d_v]

# Scaled dot-product attention
scores = (Q @ K.T) / (d_k**0.5)  # [seq_len, seq_len]
weights = F.softmax(scores, dim=-1)
attention_output = weights @ V  # [seq_len, d_v]

print(f'input_embedding {input_embedding.detach()}')

print(f'w_q weight {w_q.weight}')
print(f'w_k weight {w_k.weight}')
print(f'w_v weight {w_v.weight}')


print(f'Q {Q.detach()}')
print(f'K {K.detach()}')
print(f'V {V.detach()}')

print('Attention output')
print(attention_output.detach())


# ADD + NORM
attention_output = attention_output + input_embedding[:, :d_v]
layer_norm1 = nn.LayerNorm(d_v)
attention_output = layer_norm1(attention_output)
print(f'attention_output {attention_output}')

ffn = FFNModel(d_v)
ffn_output = ffn(attention_output)

layer_norm2 = nn.LayerNorm(d_v)
output = layer_norm2(ffn_output + attention_output)

print(output)
