import io
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import repeat


# Citation:
# @article{lample2017unsupervised,
#   title={Unsupervised Machine Translation Using Monolingual Corpora Only},
#   author={Lample, Guillaume and Conneau, Alexis and Denoyer, Ludovic and Ranzato, Marc'Aurelio},
#   journal={arXiv preprint arXiv:1711.00043},
#   year={2017}
# }


def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


def load_emb():
    src_path = input("Path to the source embeddings: ")
    tgt_path = input("Path to the target embeddings: ")
    nmax = 50000
    src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
    tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)

    return src_embeddings, src_id2word, src_word2id, tgt_embeddings, tgt_id2word, tgt_word2id


def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=8):
    #     print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    translations = []
    for i, idx in enumerate(k_best):
        #         print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
        #           print(tgt_id2word[idx])
        translations.append(tgt_id2word[idx])

    return translations


def get_src_word_count(df):
    src_word_count = {}

    for _, row in df.iterrows():
        en = row["src_word"]
        if en in src_word_count:
            src_word_count[en] = src_word_count[en] + 1
        else:
            src_word_count[en] = 1

    return src_word_count


def get_nearest_neighbours(df):
    src_word_count = get_src_word_count(df)
    src_embeddings, src_id2word, src_word2id, tgt_embeddings, tgt_id2word, tgt_word2id = load_emb()
    src_words = []
    trs = []

    for src_word, count in tqdm(src_word_count.items()):
        src_words.extend(repeat(src_word, count))
        translated = get_nn(src_word, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, K=8)
        for word in translated[:count]:
            trs.append(word)

    eval_df = {}

    eval_df["src_word"] = src_words
    eval_df["trg_word"] = trs
    MODEL = pd.DataFrame(eval_df)

    return MODEL
