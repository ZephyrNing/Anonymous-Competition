import pandas as pd
import numpy as np
import os

DATA_FILE = 'drw_data/train.parquet'
SAVE_FILE = 'drw_data/processed/train_processed.parquet'

LAG_LIST = [1, 3, 5, 10, 15, 20]
ROLL_LIST = [3, 5, 10, 15]

def generate_features(df):
    df = df.copy()
    feature_cols = df.columns.drop('label')

    feature_frames = []

    # 滞后特征
    for lag in LAG_LIST:
        lagged = df[feature_cols].shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in feature_cols]
        feature_frames.append(lagged)

    # 滚动均值和滚动标准差
    for window in ROLL_LIST:
        roll_mean = df[feature_cols].rolling(window).mean()
        roll_mean.columns = [f"{col}_rollmean{window}" for col in feature_cols]
        feature_frames.append(roll_mean)

        roll_std = df[feature_cols].rolling(window).std()
        roll_std.columns = [f"{col}_rollstd{window}" for col in feature_cols]
        feature_frames.append(roll_std)

    # 差分 & 动量
    diff = df[feature_cols] - df[feature_cols].shift(1)
    diff.columns = [f"{col}_diff1" for col in feature_cols]
    feature_frames.append(diff)

    momentum = df[feature_cols] / (df[feature_cols].shift(1) + 1e-9)
    momentum.columns = [f"{col}_momentum1" for col in feature_cols]
    feature_frames.append(momentum)

    # 合并所有特征
    features = pd.concat([df[feature_cols]] + feature_frames, axis=1)
    features['label'] = df['label']
    features = features.dropna().reset_index(drop=True)
    return features

def main():
    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    df = pd.read_parquet(DATA_FILE, engine='pyarrow')
    features = generate_features(df)
    features.to_parquet(SAVE_FILE, engine='pyarrow')
    print(f"Processed dataset saved to {SAVE_FILE}, shape: {features.shape}")

if __name__ == "__main__":
    main()
