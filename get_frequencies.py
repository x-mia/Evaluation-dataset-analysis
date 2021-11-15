import requests
from tqdm import tqdm
import time


def get_freq(src_word, trg_word, username, api_key):
    USERNAME = username
    API_KEY = api_key
    base_url = 'https://api.sketchengine.eu/bonito/run.cgi'
    data = {
        'corpname': 'preloaded/opus2_en',
        'format': 'json',
        'q': 'q[lc= "' + src_word + '"| lemma_lc= "' + src_word + '"] within opus2_sk:[lc= "' + trg_word + '"]',
        'fcrit': 'lemma',
    }
    query = requests.get(base_url + '/freqs', params=data, auth=(USERNAME, API_KEY)).json()
    query = query['fullsize']

    return query


def freq_to_df(df):
    frequencies = []

    tqdm.pandas()
    for _, row in tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True):
        time.sleep(4)
        src_word = row["src_word"]
        trg_word = row["trg_word"]
        username = input("username: ")
        api_key = input("API_KEY: ")
        frequencies.append(get_freq(src_word, trg_word, username, api_key))

    df["parallel_freqs"] = frequencies

    return df
