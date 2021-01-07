import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
import itertools
from IPython.display import display
import librosa
from joblib import dump, load
from extract_audio_features import extract_feature
import os
import argparse
import logging

logging.basicConfig(filename="memo.txt", filemode="a", format="%(asctime)s - %(message)s" ,level=logging.INFO)

def infer_audio_prob(clf, audio):
    sr = 44100
    audio_v, _ = librosa.load(audio, sr)
    feats = extract_feature(audio_v)
    audio_prob = clf.predict_proba(feats)
    return audio_prob[0]


def explain_audio_prob(audio_prob, emo_dict):
    """
    audio_prob: np [n_features,]
    """
    audio_prob_max = 0
    emo_res = []
    if len(audio_prob) != len(emo_dict.keys()):
        print("the emo_dict should be equal with audio_prob")
    else:
        audio_prob_max = np.max(audio_prob)
        emo_res = list(emo_dict.keys())[list(audio_prob).index(audio_prob_max)]
    return emo_res, audio_prob_max


def load_data():
    x_train = pd.read_csv('../data/s2e/audio_train.csv')
    x_test = pd.read_csv('../data/s2e/audio_test.csv')
    # print(x_train.shape)
    y_train = x_train['label']
    y_test = x_test['label']

    del x_train['label']
    del x_test['label']
    del x_train['wav_file']
    del x_test['wav_file']
    # print(x_test)
    return x_train, x_test, y_train, y_test





def train_ser_model(x_train, y_train, model_name):
    pass


def model_random_forest_classifier(x_train, y_train, x_test, y_test, clf_f):
    """
    Train, save model
    """
    rf_classifier = RandomForestClassifier(n_estimators=1200,
                                           min_samples_split=25)
    rf_classifier.fit(x_train, y_train)

    # Predict
    pred_probs = rf_classifier.predict_proba(x_test)

    # Results
    #display_results(y_test, pred_probs)

    with open('../pred_probas/rf_classifier_res.pkl', 'wb') as f:
        pickle.dump(pred_probs, f)

    dump(rf_classifier, clf_f)


def create_train_test_split(features, labels, test_size=0.2):
    return train_test_split(features, labels, test_size=0.20)


def infer_audio(clf_f, audio, emotion_dict):
    # load classifier
    clf = load(clf_f)

    # inference audio
    audio_prob = infer_audio_prob(clf, audio)
    emo_res, emo_prob = explain_audio_prob(audio_prob, emotion_dict)
    logging.info("emo distribution of {} : {}, predict as {} {}".format(
        os.path.basename(audio), audio_prob, emo_res, emo_prob))

def infer_audio_dir(clf_f, audio_dir, emotion_dict):
    # load classifier
    clf = load(clf_f)

    # inference audio
    a_abs = [os.path.join(audio_dir ,a) for a in os.listdir(audio_dir) if a.endswith(".wav")]
    audio_prob = infer_audio_prob(clf, audio)
    emo_res, emo_prob = explain_audio_prob(audio_prob, emotion_dict)

    print("The emo of audio:{} is {}, with prob: {}".format(audio, emo_res, emo_prob))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="../data/test_audio/Ses01F_impro01_M003.wav")
    parser.add_argument("--audio_dir", default="")
    parser.add_argument("--clf", default="../classifier/rf_classifier.pkl")

    args = parser.parse_args()
    emotion_dict = {'ang': 0,
                    'hap': 1,
                    'sad': 2,
                    'fea': 3,
                    'sur': 4,
                    'neu': 5}

    logging.info("emotions are: {}".format(emotion_dict.keys()))
    if args.audio_dir != "":
        audio_list = [os.path.join(args.audio_dir, a) for a in os.listdir(args.audio_dir) if a.endswith(".wav")]
        for audio in audio_list:
            infer_audio(args.clf, audio, emotion_dict)
    else:
        infer_audio(args.clf, args.audio, emotion_dict)
