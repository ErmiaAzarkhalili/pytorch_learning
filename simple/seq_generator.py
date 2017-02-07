import numpy as np


def seq_generator(file_name, maxlen, step):

    with open(file_name) as f:
        raw_text = f.read().lower()
    print("corpus size: {}".format(len(raw_text)))

    chars = sorted(list(set(raw_text)))
    print("corpus has {} chars".format(len(chars)))

    char_indices = {'☃': 0}
    indices_char = {0: '☃'}
    word_count = {}
    for w in raw_text:
        if w in word_count:
            word_count[w] += 1
        else:
            word_count[w] = 1

    counter = 1
    for c in chars:
        if word_count[c] > 1:
            char_indices[c] = counter
            indices_char[counter] = c
            counter += 1

    sentences = []
    next_chars = []
    for i in range(0, len(raw_text) - maxlen, step):
        sentences.append(raw_text[i: i + maxlen])
        next_chars.append(raw_text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((maxlen, len(sentences)), dtype=np.int)
    y = np.zeros((len(sentences)), dtype=np.int)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[t, i] = char_indices.get(char, 0)
        y[i] = char_indices.get(next_chars[i], 0)
    features = len(chars)

    return raw_text, sentences, char_indices, indices_char, features, X, y
