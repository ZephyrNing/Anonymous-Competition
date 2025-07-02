import pandas as pd
import lightgbm as lgb
import json
import os

with open("config.json", "r") as f:
    config = json.load(f)

train_path = config["paths"]["train_data"]
feature_path = config["paths"]["feature_file"]
model_output = config["paths"]["model_output"]

print("Loading data...")
df = pd.read_parquet(train_path)
with open(feature_path, "r") as f:
    features = [line.strip() for line in f.readlines()]

X = df[features]
y = df["label"]

print("Training LightGBM model...")
params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "verbosity": -1,
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42
}

dataset = lgb.Dataset(X, label=y)
model = lgb.train(params, dataset, num_boost_round=1000)

print("Saving model...")
os.makedirs(os.path.dirname(model_output), exist_ok=True)
model.save_model(model_output)
print("Done.")
