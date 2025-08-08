# predict_blend_properties.py

import joblib
import numpy as np
import pandas as pd
from feature_engineering import create_features  # no longer linked to training!
from nn_model import NNEnsemble

def load_all_models_all_targets():
    model_dict = {}
    for i in range(1, 11):
        model_dict[i] = {
            "lgb": joblib.load(f"models/lgb_target{i}.pkl"),
            "xgb": joblib.load(f"models/xgb_target{i}.pkl"),
            "cat": joblib.load(f"models/cat_target{i}.pkl"),
            "nn": joblib.load(f"models/nn_target{i}.pkl"),
            "meta": joblib.load(f"models/meta_target{i}.pkl"),
            "iso": joblib.load(f"models/iso_target{i}.pkl"),
            "info": joblib.load(f"models/blend_info_target{i}.pkl")
        }
    return model_dict

def predict_single_row(row_df, models_dict):
    row_feat = create_features(row_df.copy())
    row_feat.fillna(0, inplace=True)

    predictions = []

    for target_num in range(1, 11):
        models = models_dict[target_num]
        selected_features = models["info"]["selected_features"]
        X_input = row_feat[selected_features]  # <-- fixed: keep as DataFrame with column names


        base_preds = np.vstack([
            models["lgb"].predict(X_input),
            models["xgb"].predict(X_input),
            models["cat"].predict(X_input),
            models["nn"].predict(X_input)
        ]).T

        meta_pred = models["meta"].predict(base_preds)
        y_mean, y_std = models["info"]["mean"], models["info"]["std"]
        pred_raw = meta_pred * y_std + y_mean
        calibrated = models["iso"].transform(pred_raw)

        blend_baseline = sum(
            row_df[f"Component{j}_fraction"].values[0] * row_df[f"Component{j}_Property{target_num}"].values[0]
            for j in range(1, 6)
        )

        alpha = models["info"]["alpha"]
        final_pred = alpha * calibrated + (1 - alpha) * blend_baseline
        final_pred = float(np.clip(final_pred, models["info"]["min"], models["info"]["max"]))

        predictions.append(final_pred)

    return predictions
