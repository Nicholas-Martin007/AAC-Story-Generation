from collections import defaultdict

# Corpus
corpus = [
    'Anak-anak membaca cerita sosial untuk belajar tentang teman-teman mereka dan cerita sosial lainnya',
    'Kopi pagi membuat hari terasa hangat, dan kopi pagi itu selalu menyenangkan',
    'Teknologi baru bisa mengubah hidup menjadi lebih mudah, dan teknologi baru itu sangat membantu',
]

#  Initialize
candidates = set()
for sentence in corpus:
    for word in sentence.split():
        for i in range(len(word)):
            for j in range(i + 1, len(word) + 1):
                candidates.add(word[i:j])
candidates = list(candidates)

# Assign
prob = {token: 1 / len(candidates) for token in candidates}


# Tokenize word
def tokenize_word(word, candidates_prob):
    tokens = []
    i = 0
    while i < len(word):
        match = ''
        for j in range(i + 1, len(word) + 1):
            sub = word[i:j]
            if sub in candidates_prob and len(sub) > len(match):
                match = sub
        if match == '':
            match = word[i]
        tokens.append(match)
        i += len(match)
    return tokens


# Step 4:  prune
num_iterations = 5
for _ in range(num_iterations):
    counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence.split():
            tokens = tokenize_word(word, prob)
            for t in tokens:
                counts[t] += 1
    total = sum(counts.values())
    for t in counts:
        prob[t] = counts[t] / total
    threshold = 0.02
    prob = {t: p for t, p in prob.items() if p >= threshold}

# Step 5: Build vocabulary mapping subword -> ID
vocab = {
    token: idx for idx, token in enumerate(sorted(prob.keys()))
}

# Step 6: Tokenize corpus using final unigram probabilities
tokenized_corpus = []
for sentence in corpus:
    tokenized_sentence = []
    for word in sentence.split():
        tokenized_sentence.extend(tokenize_word(word, prob))
    tokenized_corpus.append(tokenized_sentence)

# Step 7: Convert tokenized corpus to IDs
numerical_corpus = []
for sentence_tokens in tokenized_corpus:
    numerical_corpus.append([vocab[t] for t in sentence_tokens])

# Step 8: Tokenize your input sentence
input_text = 'Saya mau menciptakan sebuah kisah sosial yang baru'
tokenized_input = []
for word in input_text.split():
    tokenized_input.extend(
        tokenize_word(word, prob)
    )  # lowercase to match corpus

input_ids = [vocab[t] for t in tokenized_input if t in vocab]

# Print results
print('Final subword probabilities:')
print(prob)

print('\nTokenized input sentence:', tokenized_input)
