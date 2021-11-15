import numpy as np


# from matplotlib import pyplot as plt
# import seaborn as sns

def get_cat_weights(df):
    ctg_weights = []

    for _, row in df.iterrows():
        ctg = row["category"]
        if ctg == "A":
            ctg_weights.append(1)
        if ctg == "B":
            ctg_weights.append(0.8)
        if ctg == "C":
            ctg_weights.append(0.3)
        if ctg == "D":
            ctg_weights.append(0.2)
        if ctg == "E":
            ctg_weights.append(0.2)
        if ctg == "F":
            ctg_weights.append(0.1)
        if ctg == "G":
            ctg_weights.append(0.2)
        if ctg == "H":
            ctg_weights.append(0.1)
        if ctg == "I":
            ctg_weights.append(0.8)
        if ctg == "J":
            ctg_weights.append(0.6)

    df["ctg_weights"] = ctg_weights

    # Plotting the number of word pairs in each category:
    # a = len(df[df["category"] == "A"])
    # b = len(df[df["category"] == "B"])
    # c = len(df[df["category"] == "C"])
    # d = len(df[df["category"] == "D"])
    # e = len(df[df["category"] == "E"])
    # f = len(df[df["category"] == "F"])
    # g = len(df[df["category"] == "G"])
    # h = len(df[df["category"] == "H"])
    # i = len(df[df["category"] == "I"])
    # j = len(df[df["category"] == "J"])

    # names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    # values = [a, b, c, d, e, f, g, h, i, j]
    # my_colors = 'rgbkymc'
    #
    # plt.bar(names, values)
    # plt.xlabel("Categories")
    # plt.ylabel("The number of word pairs")
    # plt.savefig('Categories.png', dpi=150)

    return df


def NormalizeFreqs(freqs_log):
    normalized_freqs = (freqs_log - np.min(freqs_log)) / (np.max(freqs_log) - np.min(freqs_log))

    return normalized_freqs


def get_scaled_freqs(df):
    freqs = df["parallel_freqs"]
    # If scaling without logarithm then comment following row:
    freqs_log = np.log(freqs + 0.001)
    scaled_x = list(NormalizeFreqs(freqs_log))
    df["scaled_freqs"] = scaled_x

    # Plotting Zipf's curve:
    # freqs_sorted = freqs.sort_values(ascending=False).reset_index(drop=True)
    # freqs_sorted.plot(loglog=True)
    # plt.savefig("zipfs_law.png", dpi=150)

    return df


def clip_sim(df):
    sim = list(df.iloc[:, 5])
    sim_ar = np.array(sim)
    sml = np.clip(sim_ar, 0, 1)
    clip_simp = list(sml)

    return clip_simp


def get_weights(df):
    df["clip_sim"] = clip_sim(df)

    # Plotting cliped similarities:
    # sns.kdeplot(df.clip_sim)
    # plt.xlim(0, 1)
    # plt.xlabel("Similarity")
    # plt.savefig('similarity.png', dpi=150)

    weights = []
    for _, row in df.iterrows():
        sim = row["clip_sim"]
        freq = row["scaled_freqs"]
        ctg = row["ctg_weights"]
        weight = sim * freq * ctg
        weights.append(weight)

    df["weights"] = weights

    # Plotting overlapping histograms of weights distribution in each category:
    # sns.kdeplot(df.clip_weights, hue=df.category)
    # plt.xlim(0, 1)
    # plt.xlabel("weights")
    # plt.savefig('cat_vs_weights.png', dpi=150)

    return df
