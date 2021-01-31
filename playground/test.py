import re
import os
import pickle
import unicodedata
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



if __name__ == '__main__':
    regex = re.compile(r'(.*)_(.*)_(.*)', re.IGNORECASE)

    df = pd.read_csv("audio_features.csv")
    df["gender_e"] = [regex.match(wav_file).group(3)[0] for wav_file in df["wav_file"]]
    df["gender"] = df["gender_e"].map({"F": 0, "M":1})
    del df["gender_e"]
    df.to_csv("audio_features_gender.csv")