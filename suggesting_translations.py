import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import fasttext
import fasttext.util


def suggest_translation(df):
    df['sug_trs'] = pd.Series(dtype='str')

    for i, row in df.iterrows():
        cat = row["category"]
        src_word = row["src_word"]
        print(src_word)
        trg_word = row["trg_word"]
        print(trg_word)

        if cat == "A":
            df.at[i, "sug_trs"] = trg_word
        else:
            category = input("translation: ")
            df.at[i, "sug_trs"] = category

        return df


def get_similarity(trg_word, suggestion, ft):
    a = np.reshape(ft.get_word_vector(trg_word), (1, -1))
    b = np.reshape(ft.get_word_vector(suggestion), (1, -1))
    cos_sim = cosine_similarity(a, b)[0][0]

    return cos_sim


def sim_to_df(df, ft):
    df = suggest_translation(df)

    similarities = []

    tqdm.pandas()
    for _, row in tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True):
        trg_word = row["trg_word"]
        suggestion = row["sug_trs"]
        similarities.append(get_similarity(trg_word, suggestion, ft))
    df["similarity"] = similarities

    return df
