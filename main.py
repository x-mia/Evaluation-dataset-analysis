from dataset_pre_processing import get_df
from get_frequencies import freq_to_df
from category_annotation import input_category
from suggesting_translations import sim_to_df
from get_weights import get_cat_weights, get_scaled_freqs, get_weights
from testing_dataset import get_nearest_neighbours
from comparison import compare
import fasttext
import fasttext.util
import pandas as pd


# !wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sk.300.bin.gz
# !gunzip cc.sk.300.bin.gz

def main():
    data = input("Path to your data file: ")
    df = get_df(data)
    df = freq_to_df(df)
    df = input_category(df)
    ft = fasttext.load_model(input("Path to your model bin file: "))
    df = sim_to_df(df, ft)
    df = get_cat_weights(df)
    df = get_scaled_freqs(df)
    df = get_weights(df)
    # df.to_csv("df.csv", index=False)
    model_df = get_nearest_neighbours(df)
    # model_df.to_csv("model_df.csv", index=False)
    accuracy, weighted_accuracy = compare(df, model_df)

    return df, model_df


if __name__ == "__main__":
    main()
