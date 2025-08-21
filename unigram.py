from collections import defaultdict

# Example corpus
corpus = ['saya mau kopi', 'makan itu kenyang']

# Step 1: Initialize candidate subwords (all substrings)
candidates = set()
for sentence in corpus:
    for word in sentence.split():
        for i in range(len(word)):
            for j in range(i + 1, len(word) + 1):
                candidates.add(word[i:j])
candidates = list(candidates)

# Step 2: Assign initial probabilities (uniform)
prob = {token: 1 / len(candidates) for token in candidates}


# Step 3: Tokenize word using current candidates
def tokenize_word(word, candidates_prob):
    tokens = []
    i = 0
    while i < len(word):
        # Find the longest matching candidate
        match = ''
        for j in range(i + 1, len(word) + 1):
            sub = word[i:j]
            if sub in candidates_prob and len(sub) > len(match):
                match = sub
        tokens.append(match)
        i += len(match)
    return tokens


# Step 4: Iteratively prune low probability subwords
num_iterations = 5
for _ in range(num_iterations):
    # Count usage of each candidate
    counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence.split():
            tokens = tokenize_word(word, prob)
            for t in tokens:
                counts[t] += 1

    # Recompute probabilities
    total = sum(counts.values())
    for t in counts:
        prob[t] = counts[t] / total

    # Prune low probability tokens randomly
    threshold = 0.05  # keep frequent subwords
    prob = {t: p for t, p in prob.items() if p >= threshold}

# Step 5: Tokenize corpus using final unigram probabilities
tokenized_corpus = []
for sentence in corpus:
    tokenized_sentence = []
    for word in sentence.split():
        tokenized_sentence.extend(tokenize_word(word, prob))
    tokenized_corpus.append(tokenized_sentence)

print('Final subword probabilities:', prob)
print('Tokenized corpus:', tokenized_corpus)
