"""
This script preprocesses data and prepares data to be actually used in training
"""
import re
import os
import pickle
import unicodedata
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(filename="memo_1.txt", level=logging.INFO)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """
    Lowercase, trim, and remove non-letter characters
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def transcribe_sessions():
    file2transcriptions = {}
    useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)
    transcript_path = '/home/Data/IEMOCAP_session_only/Session{}/dialog/transcriptions/'
    for sess in range(1, 6):
        transcript_path_i = transcript_path.format(sess)
        for f in os.listdir(transcript_path_i):
            with open('{}{}'.format(transcript_path_i, f), 'r') as f:
                all_lines = f.readlines()
            for l in all_lines:
                logging.info(l)
                audio_code = useful_regex.match(l).group()
                transcription = l.split(':')[-1].strip()
                # assuming that all the keys would be unique and hence no `try`
                file2transcriptions[audio_code] = transcription
    with open('../data/t2e/audiocode2text.pkl', 'wb') as file:
        pickle.dump(file2transcriptions, file)
    return file2transcriptions


def prepare_text_data(audiocode2text):
    # Prepare text data
    df = pd.read_csv('../data/pre-processed/audio_features.csv')
    # Delete xxx and oth
    df = df[df['label'].isin([0, 1, 2, 3, 4, 5, 6, 7])]

    # Delete fea
    df = df[df['label'].isin([0, 1, 2, 3, 4, 5, 6, 7])]

    print(df.shape)
    # change 7 to 6
    #df['label'] = df['label'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5})

    # change 7 to 4
    df['label'] = df['label'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5})
    # append gender
    use_gender = False
    if use_gender:
        regex_gender = re.compile(r'(.*)_(.*)_(.*)', re.IGNORECASE)
        df["gender_e"] = [regex_gender.match(wav_file).group(3)[0] for wav_file in df["wav_file"]]
        df["gender"] = df["gender_e"].map({"F": 0, "M":1})
        del df["gender_e"]
        df.head()
        df.to_csv('../data/no_sample_df.csv')

    # oversample fear and sur
    fear_df = df[df['label'] == 3]
    for i in range(30):
        df = df.append(fear_df)

    sur_df = df[df['label'] == 4]
    for i in range(10):
        df = df.append(sur_df)

    df.to_csv('../data/modified_df.csv')
    print(df)

    # furthermore reduce emo number ?
    emotion_dict = {'ang': 0,
                    'hap': 1,
                    'sad': 2,
                    'fea': 3,
                    'sur': 4,
                    'neu': 5
                    }
    # original
    # emotion_dict = {'ang': 0,
    #                 'hap': 1,
    #                 'exc': 2,
    #                 'sad': 3,
    #                 'fru': 4,
    #                 'fea': 5,
    #                 'sur': 6,
    #                 'neu': 7,
    #                 'xxx': 8,
    #                 'oth': 8}

    scalar = MinMaxScaler()
    df[df.columns[2:]] = scalar.fit_transform(df[df.columns[2:]])

    # Save feats stats
    feats_stats = pd.DataFrame()
    feats_stats["min"] = scalar.data_min_
    feats_stats["max"] = scalar.data_max_
    feats_stats = feats_stats.transpose()
    feats_stats_file = "../data/s2e/feats_stats.csv"
    feats_stats.to_csv(feats_stats_file, index=False)


    x_train, x_test = train_test_split(df, test_size=0.20)
    x_train.to_csv('../data/s2e/audio_train.csv', index=False)
    x_test.to_csv('../data/s2e/audio_test.csv', index=False)

    text_train = pd.DataFrame()
    text_train['wav_file'] = x_train['wav_file']
    text_train['label'] = x_train['label']
    text_train['transcription'] = [normalizeString(audiocode2text[code])
                                   for code in x_train['wav_file']]

    text_test = pd.DataFrame()
    text_test['wav_file'] = x_test['wav_file']
    text_test['label'] = x_test['label']
    text_test['transcription'] = [normalizeString(audiocode2text[code])
                                  for code in x_test['wav_file']]

    text_train.to_csv('../data/t2e/text_train.csv', index=False)
    text_test.to_csv('../data/t2e/text_test.csv', index=False)

    print(text_train.shape, text_test.shape)


def main():
    prepare_text_data(transcribe_sessions())


if __name__ == '__main__':
    main()
