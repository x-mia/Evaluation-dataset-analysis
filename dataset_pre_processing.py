import pandas as pd


def load_text(data):
    with open(data, "r", encoding="utf-8") as f:
        data = f.readlines()

    return data


def clean_data(data):
    splitted_els = []
    cleaned_data = []

    for el in data:
        split_el = el.split("\n")
        splitted_els.append(split_el[0])

    for x in splitted_els:
        cleaned_x = x.split("\t")
        cleaned_data.append(cleaned_x)

    return cleaned_data


def dict_to_df(cleaned_data):
    dict_df = {}
    src_words = []
    trg_words = []

    for i in cleaned_data:
        src_words.append(i[0])
        trg_words.append(i[1])

    dict_df["src_word"] = src_words
    dict_df["trg_word"] = trg_words

    df = pd.DataFrame(dict_df)

    return df


def get_df(data):
    data = load_text(data)
    cleaned_data = clean_data(data)
    df = dict_to_df(cleaned_data)

    return df
