list_sentences = [
    'Anak-anak membaca cerita sosial untuk belajar tentang teman-teman mereka dan cerita sosial lainnya',
    'Kopi pagi membuat hari terasa hangat, dan kopi pagi itu selalu menyenangkan',
    'Teknologi baru bisa mengubah hidup menjadi lebih mudah, dan teknologi baru itu sangat membantu',
]


from collections import defaultdict

word_frequencies = defaultdict(int)  # storage
char_frequencies = defaultdict(int)

word_char_list = {}

pairs = defaultdict(int)
merge_result = {}

# sentence -> word
for sentence in list_sentences:
    for word in sentence.split():
        word_frequencies[word.lower()] += 1

# word -> char
for word in word_frequencies.keys():
    word_char_list[word] = list(word)

# word_char list -> char
for word, freq in word_frequencies.items():
    for char in word_char_list[word]:
        char_frequencies[char] += freq


def get_bpe_pairs():
    pairs.clear()
    for word, char_list in word_char_list.items():
        for i in range(len(char_list) - 1):
            pairs[char_list[i], char_list[i + 1]] += (
                word_frequencies[word]
            )


def merge(pair):
    n_gram = ''.join(pair)
    for word, chars in word_char_list.items():
        new_chars = []
        i = 0
        while i < len(chars):
            if (
                i < len(chars) - 1
                and chars[i] == pair[0]
                and chars[i + 1] == pair[1]
            ):
                new_chars.append(n_gram)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        word_char_list[word] = new_chars


if __name__ == '__main__':
    vocab_size = 50  # our vocab size
    for i in range(vocab_size):
        get_bpe_pairs()
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        print(
            f'Merge {i + 1}: {best_pair} -> {pairs[best_pair]}'
        )
        merge(best_pair)

    for word, chars in word_char_list.items():
        print(word, '||', chars)

    tokenized_corpus = []

    for sentence in ['saya mau kopi', 'makan itu kenyang']:
        tokenized_sentence = []
        for word in sentence.split():
            word_lower = word.lower()
            tokenized_sentence.extend(
                word_char_list.get(word_lower, list(word_lower))
            )
        tokenized_corpus.append(tokenized_sentence)

    for i, tokens in enumerate(tokenized_corpus):
        print(f'Sentence {i}: {tokens}')

    # Step 1: Build vocabulary from all subwords
    vocab = {}
    current_id = 0
    for chars in word_char_list.values():
        for token in chars:
            if token not in vocab:
                vocab[token] = current_id
                current_id += 1

    # Step 2: Convert tokenized corpus to numerical IDs
    numerical_corpus = []
    for sentence in list_sentences:
        token_ids = []
        for word in sentence.split():
            word_lower = word.lower()
            subwords = word_char_list.get(
                word_lower, list(word_lower)
            )
            token_ids.extend(
                [vocab[token] for token in subwords]
            )
        numerical_corpus.append(token_ids)

    # Print results
    print('Vocabulary:', vocab)
    for i, seq in enumerate(numerical_corpus, 1):
        print(f'Sentence {i} IDs: {seq}')
