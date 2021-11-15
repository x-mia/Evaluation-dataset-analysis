import pandas as pd


def compare(df, model_df):
    # if not sorted or not columns with same name:
    # df = df.sort_values('src_word')
    # df = df.reset_index(drop=True)
    # df = df.rename(columns={"sk": "trg_word"})

    # find non matching rows:
    df_non_match = pd.merge(df, model_df, how='outer', indicator=True, on=['trg_word', 'src_word'])
    df_non_match = df_non_match[(df_non_match._merge != 'both')]
    print(df_non_match)

    # find matching rows:
    df_matching_rows = pd.merge(df, model_df, how='left', indicator=True, on=['trg_word', 'src_word'])
    match = (df_matching_rows["_merge"] == 'both')
    print(match)

    # mean:
    accuracy = len(match) / len(df_matching_rows)
    print(accuracy)

    # weighted mean:
    w = df_matching_rows["weights"]
    weighted_accuracy = ((match * w).sum() / w.sum())
    print(weighted_accuracy)


    return accuracy, weighted_accuracy
