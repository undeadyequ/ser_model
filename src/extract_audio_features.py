"""
This script extract features from existing audio vectors
"""

import pickle
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import math


def add_session_data(df_features, labels_df, emotion_dict, audio_vectors_path, sess, columns):
    audio_vectors = pickle.load(open(audio_vectors_path, 'rb'))
    for index, row in tqdm(labels_df[labels_df['wav_file'].str.contains(
            'Ses0{}'.format(sess))].iterrows()):
        try:
            wav_file_name = row['wav_file']
            label = emotion_dict[row['emotion']]
            y = audio_vectors[wav_file_name]

            feature_list = [wav_file_name, label]  # wav_file, label
            sig_mean = np.mean(abs(y))
            feature_list.append(sig_mean)  # sig_mean
            feature_list.append(np.std(y))  # sig_std

            rmse = librosa.feature.rms(y + 0.0001)[0]
            feature_list.append(np.mean(rmse))  # rmse_mean
            feature_list.append(np.std(rmse))  # rmse_std

            silence = 0
            for e in rmse:
                if e <= 0.4 * np.mean(rmse):
                    silence += 1
            silence /= float(len(rmse))
            feature_list.append(silence)  # silence

            y_harmonic = librosa.effects.hpss(y)[0]
            feature_list.append(np.mean(y_harmonic) * 1000)  # harmonic (scaled by 1000)

            # based on the pitch detection algorithm mentioned here:
            # http://access.feld.cvut.cz/view.php?cisloclanku=2009060001
            cl = 0.45 * sig_mean
            center_clipped = []
            for s in y:
                if s >= cl:
                    center_clipped.append(s - cl)
                elif s <= -cl:
                    center_clipped.append(s + cl)
                elif np.abs(s) < cl:
                    center_clipped.append(0)
            p3 = time.time()
            #auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
            pitch, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            pitch = [0 if math.isnan(p) else p for p in pitch]
            p4 = time.time()
            print("audio size: {}, pitch:{}".format(len(y)/44100.0, (p4-p3)))

            feature_list.append(np.mean(pitch))
            feature_list.append(np.std(pitch))
            #feature_list.append(1000 * np.max(auto_corrs)/len(auto_corrs))  # auto_corr_max (scaled by 1000)
            #feature_list.append(np.std(auto_corrs))  # auto_corr_std

            df_features = df_features.append(pd.DataFrame(feature_list, index=columns).transpose(), ignore_index=True)
        except Exception as e:
            print('Some exception occured: {}'.format(e))
    return df_features


def main():
    emotion_dict = {'ang': 0, 'hap': 1, 'exc': 2, 'sad': 3, 'fru': 4, 'fea': 5,
                    'sur': 6, 'neu': 7, 'xxx': 8, 'oth': 8}

    data_dir = '../data/pre-processed/'
    labels_path = '{}df_iemocap_test.csv'.format(data_dir)
    audio_vectors_path = '{}audio_vectors_'.format(data_dir)
    #columns = ['wav_file', 'label', 'sig_mean', 'sig_std', 'rmse_mean',
    #           'rmse_std', 'silence', 'harmonic', 'auto_corr_max',
    #           'auto_corr_std']
    columns = ['wav_file', 'label', 'sig_mean', 'sig_std', 'rmse_mean',
               'rmse_std', 'silence', 'harmonic', 'pitch_mean',
               'pitch_std']
    df_features = pd.DataFrame(columns=columns)
    labels_df = pd.read_csv(labels_path)
    for sess in range(1, 2):
        df_features = add_session_data(df_features, labels_df, emotion_dict,
                         '{}{}.pkl'.format(audio_vectors_path, sess), sess, columns)
    df_features.to_csv('../data/pre-processed/audio_features.csv', index=False)

if __name__ == '__main__':
    main()
