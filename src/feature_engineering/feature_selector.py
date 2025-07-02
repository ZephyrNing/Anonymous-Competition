import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_selection import VarianceThreshold
import shap
import os

config = {
    "feature_selection": {
        "use_variance": True,
        "use_lgbm": True,
        "use_shap": False,
        "top_k_lgbm": 300,
        "top_k_shap": 0,
        "sample_size": 10000
    },
    "paths": {
        "data_file": "drw_data/train.parquet",
        "output_file": "drw_data/selected_features.txt"
    }
}

def load_data():
    df = pd.read_parquet(config["paths"]["data_file"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    x_cols = [c for c in df.columns if c.startswith("X")]
    df = df[x_cols + ['label']]
    if config["feature_selection"]["sample_size"]:
        df = df.iloc[:config["feature_selection"]["sample_size"]]
    return df

def apply_variance_filter(X):
    X_cleaned = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    selector = VarianceThreshold(threshold=1e-4)
    X_filtered = selector.fit_transform(X_cleaned)
    retained = X_cleaned.columns[selector.get_support()]
    return X_cleaned[retained]

def apply_lgbm_selection(X, y):
    model = lgb.LGBMRegressor(
        n_estimators=100,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0
    )
    model.fit(X, y)
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    top_features = imp_df.sort_values('importance', ascending=False)['feature'].head(config["feature_selection"]["top_k_lgbm"]).tolist()
    return model, top_features

def apply_shap_selection(model, X, top_k, batch_size=1000):
    explainer = shap.TreeExplainer(model.booster_, feature_perturbation='tree_path_dependent')
    shap_sums = np.zeros(X.shape[1])
    n_batches = int(np.ceil(len(X) / batch_size))

    for i in range(n_batches):
        batch = X.iloc[i*batch_size : (i+1)*batch_size]
        shap_vals = explainer.shap_values(batch)
        shap_sums += np.abs(shap_vals).sum(axis=0)

    shap_mean = shap_sums / len(X)
    top_indices = np.argsort(shap_mean)[-top_k:]
    return X.columns[top_indices].tolist()

def save_feature_list(features):
    output_path = config["paths"]["output_file"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for feat in features:
            f.write(f"{feat}\n")

def main():
    df = load_data()
    X, y = df.drop(columns='label'), df['label']

    print("done!")

    if config["feature_selection"]["use_variance"]:
        X = apply_variance_filter(X)
        print("Var done!")

    X_lgbm = X.copy()

    if config["feature_selection"]["use_lgbm"]:
        model, lgbm_top_features = apply_lgbm_selection(X_lgbm, y)
        X_lgbm = X_lgbm[lgbm_top_features]
        print("LGBM done!")

    if config["feature_selection"]["use_shap"]:
        shap_top_features = apply_shap_selection(model, X_lgbm, config["feature_selection"]["top_k_shap"])
        selected_features = shap_top_features
    else:
        selected_features = lgbm_top_features

    save_feature_list(selected_features)

if __name__ == "__main__":
    main()
