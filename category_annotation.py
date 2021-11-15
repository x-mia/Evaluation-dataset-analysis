import pandas as pd


def input_category(df):
    df['category'] = pd.Series(dtype='str')
    for i, row in df.iterrows():
        src_word = row["src_word"]
        print(src_word)
        trg_word = row["trg_word"]
        print(trg_word)

        category = input("category: ")
        df.at[i, "category"] = category

    return df
