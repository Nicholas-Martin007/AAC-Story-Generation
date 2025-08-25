from collections import defaultdict

# Corpus
corpus = [
    'Anak-anak membaca cerita sosial untuk belajar tentang teman-teman mereka dan cerita sosial lainnya',
    'Kopi pagi membuat hari terasa hangat, dan kopi pagi itu selalu menyenangkan',
    'Teknologi baru bisa mengubah hidup menjadi lebih mudah, dan teknologi baru itu sangat membantu',
]

# Step 1: start with lowercase characters as initial vocab
vocab = set()
for sentence in corpus:
    for word in sentence.split():
        for char in word.lower():
            vocab.add(char)
vocab = list(vocab)

# Step 2: count substring frequencies (all lowercase)
subword_freqs = defaultdict(int)
for sentence in corpus:
    for word in sentence.split():
        chars = list(word.lower())
        for i in range(len(chars)):
            for j in range(i + 1, len(chars) + 1):
                subword_freqs[''.join(chars[i:j])] += 1

# Step 3: grow vocab up to target size
target_vocab_size = 50
while len(vocab) < target_vocab_size and subword_freqs:
    candidate = max(subword_freqs.items(), key=lambda x: x[1])
    if candidate[0] not in vocab:
        vocab.append(candidate[0])
    del subword_freqs[candidate[0]]

# Step 4: map vocab to IDs
vocab_to_id = {token: idx for idx, token in enumerate(vocab)}


# Step 5: tokenize word using longest-match
def tokenize_word(word, vocab_set):
    tokens = []
    i = 0
    while i < len(word):
        match = ''
        for j in range(i + 1, len(word) + 1):
            sub = word[i:j]
            if sub in vocab_set and len(sub) > len(match):
                match = sub
        if match == '':
            match = word[i]  # fallback to single character
        tokens.append(match)
        i += len(match)
    return tokens


# Step 6: tokenize corpus and convert to IDs
tokenized_ids = []
for sentence in corpus:
    tok_sentence = []
    for word in sentence.split():
        tokens = tokenize_word(word.lower(), set(vocab))
        tok_sentence.extend([vocab_to_id[t] for t in tokens])
    tokenized_ids.append(tok_sentence)

# Step 7: tokenize your input sentence
input_text = 'Saya mau menciptakan sebuah kisah sosial yang baru'
tokenized_input = []
tokenized_input_ids = []
for word in input_text.split():
    tokens = tokenize_word(word.lower(), set(vocab))
    tokenized_input.extend(tokens)
    tokenized_input_ids.extend([vocab_to_id[t] for t in tokens])

# Output
print('Vocabulary:', vocab_to_id)
print('Tokenized corpus (IDs):', tokenized_ids)
print('\nTokenized input sentence:', tokenized_input)
print('Input IDs:', tokenized_input_ids)
