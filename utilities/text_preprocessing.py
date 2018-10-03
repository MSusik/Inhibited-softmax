import json
import numpy as np
import re
import torchwordemb
import tqdm

from gensim.parsing.preprocessing import remove_stopwords


def embed_reviews(text):
    text = re.sub("<br />", " ", text)
    cleared = re.sub("[^\w\s\d]+", "", remove_stopwords(text.lower()))
    return cleared


def fill_embedded_matrix(texts):

    vocab, vec = torchwordemb.load_glove_text(
        './movies/glove.6B/glove.6B.300d.txt'
    )
    embedded_matrix = np.zeros((texts.shape[0], 100, 300), dtype=np.float16)
    row_ = np.zeros((100, 300), dtype=np.float16)

    ## TO change row to texts
    split_texts = texts.cleared_text.str.split()
    for i, split_text in enumerate(split_texts):
        index = 0
        filled = 0
        while filled < 100 and index < len(split_text):
            if split_text[index] in vocab:
                row_[filled, :] = vec[vocab[split_text[index]]]
                filled += 1
            index += 1
        # Pad from left
        embedded_matrix[i, 100 - filled:, :] = row_[:filled, :]

    return embedded_matrix


def preprocess_sentiment_df(texts):
    """
    :param texts: df with 'text' column containing articles
    :param path: string - where to save the matrix
    :return:
    """

    texts['cleared_text'] = texts['text'].apply(embed_reviews)
    texts['length'] = texts.cleared_text.apply(lambda x: len(x.split()))
    return texts


def get_indices_from_text(table, indices, for_embedding, thresholded=5044):

    for row in table.iterrows():
        if row[1].name % 10000 == 0:
            print(row[1].name)
        row_ = np.zeros((400), dtype=np.int)

        index = 0
        for word in row[1].cleared_text.split(' '):
            if index == 400:
                break
            if word in indices:
                row_[index] = indices[word]
            else:
                row_[index] = thresholded
            index += 1

        for_embedding[row[1].name, 400 - index:] = row_[:index]

    return for_embedding


def get_occurences(texts, threshold=77):
    occurences = {}

    for text in tqdm.tqdm(texts[:25000].cleared_text):
        for word in text.split(' '):
            if word:
                if word in occurences:
                    occurences[word] += 1
                else:
                    occurences[word] = 1

    thresholded = len({k:v for k,v in occurences.items() if v > threshold})
    print("Lefting {} words".format(thresholded))

    occurences = {k: v for k, v in occurences.items() if v > 77}
    indices = {k: i for i, k in enumerate(occurences)}
    json.dump(indices, open('indices.json', 'w'))

    for_embedding = np.zeros((50000, 400), dtype=np.int) + thresholded

    get_indices_from_text(texts, indices, for_embedding)

    train = for_embedding[:25000, :]
    test = for_embedding[25000:, :]

    np.save(open('./movies/train_e.npy', "wb"), train)
    np.save(open('./movies/test_e.npy', "wb"), test)

    return for_embedding


def preprocess_other_dataset(texts, path, thresholded):
    length = texts.shape[0]
    for_embedding = np.zeros((length, 400), dtype=np.int) + thresholded

    indices = json.load(open('indices.json', 'r'))
    for_embedding = get_indices_from_text(
        texts,
        indices,
        for_embedding,
        thresholded
    )

    np.save(open(path, "wb"), for_embedding)

    return for_embedding
