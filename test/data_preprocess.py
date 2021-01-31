import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# 1. Data norm
    # Out of range in test data

def test_data_norm(train_feats_stats_file):
    train_feats_stats = pd.read_csv(train_feats_stats_file)
    scalar = MinMaxScaler()
    scalar.fit(train_feats_stats)

    """
    min 0.00058207207,0.0007455582000000001,0.0007287584599999999,0.000106959604,0.0,-0.8698079618625343,0.0,0.0
    max 0.2853865,0.37650034,0.34655790000000003,0.18235283,0.7687861271676301,4.240148700773717,378.5284629813393,614.5640833762186

    """
    feats = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]])
    feats_norm = scalar.transform(feats)
    print(feats_norm)


if __name__ == '__main__':
    train_feats_stats_file = "../data/s2e/feats_stats.csv"
    test_data_norm(train_feats_stats_file)