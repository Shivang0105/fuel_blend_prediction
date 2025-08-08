import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import shap

from nn_model import NNEnsemble
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import joblib
import os
os.makedirs("models", exist_ok=True)

# Data Loading
train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")
sample = pd.read_csv("datasets/sample_submission.csv")

test_ids = test["ID"]
train.drop(columns=["ID"], errors="ignore", inplace=True)
test.drop(columns=["ID"], errors="ignore", inplace=True)

target_cols = [f"BlendProperty{i}" for i in range(1, 11)]

# Feature Engineering
def create_features(df):
    df = df.copy()

    # Weighted properties (strong baseline feature for blending)
    for i in range(1, 11):
        df[f'BlendWeighted_Property{i}'] = sum(
            df[f'Component{j}_fraction'] * df[f'Component{j}_Property{i}'] for j in range(1, 6)
        )

    # Residuals from the weighted blend
    for i in range(1, 11):
        blend = df[f'BlendWeighted_Property{i}']
        for j in range(1, 6):
            df[f'Residual_Component{j}_Prop{i}'] = df[f'Component{j}_Property{i}'] - blend

    # Component-wise stats
    for j in range(1, 6):
        props = [f'Component{j}_Property{i}' for i in range(1, 11)]
        df[f'Component{j}_mean'] = df[props].mean(axis=1)
        df[f'Component{j}_std'] = df[props].std(axis=1)

    # Interaction between fraction and property
    for j in range(1, 6):
        for i in range(1, 11):
            df[f'Frac{j}_x_Prop{i}'] = df[f'Component{j}_fraction'] * df[f'Component{j}_Property{i}']

    # Property-wise stats
    for i in range(1, 11):
        props = [f"Component{j}_Property{i}" for j in range(1, 6)]
        df[f'Property{i}_max'] = df[props].max(axis=1)
        df[f'Property{i}_min'] = df[props].min(axis=1)
        df[f'Property{i}_std'] = df[props].std(axis=1)

    # Fraction stats
    frac_cols = [f"Component{i}_fraction" for i in range(1, 6)]
    df["frac_sum"] = df[frac_cols].sum(axis=1)
    df["frac_max"] = df[frac_cols].max(axis=1)
    df["frac_min"] = df[frac_cols].min(axis=1)
    df["frac_std"] = df[frac_cols].std(axis=1)

    return df

# Prepair data
X_full = create_features(train.drop(columns=target_cols))
y = train[target_cols].copy()
X_test_full = create_features(test)

X_full, X_test_full = X_full.align(X_test_full, join='outer', axis=1, fill_value=0)

X_full.fillna(0, inplace=True)
X_test_full.fillna(0, inplace=True)

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_full)
X_test_scaled = scaler.transform(X_test_full)
X_df = pd.DataFrame(X_scaled, columns=X_full.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X_full.columns)

# # Neural Network Ensemble Class
# class NNEnsemble:
#     def __init__(self, n_models=3, random_state=None):

#         self.models = []
#         self.n_models = n_models
#         self.random_state = random_state

#     def fit(self, X, y):

#         for i in range(self.n_models):
#             model = MLPRegressor(
#                 hidden_layer_sizes=(256, 128, 64),
#                 activation='relu',
#                 alpha=1e-3,
#                 learning_rate='adaptive',
#                 learning_rate_init=0.001,
#                 max_iter=500,
#                 early_stopping=True,
#                 n_iter_no_change=20,
#                 validation_fraction=0.1,
#                 random_state=self.random_state + i if self.random_state is not None else None,
#                 solver='adam',
#                 beta_1=0.9,
#                 beta_2=0.999,
#                 epsilon=1e-8,
#                 tol=1e-4,
#                 verbose=0,
#                 batch_size=16
#             )
#             model.fit(X, y)
#             self.models.append(model)

#     def predict(self, X):
#         preds = np.array([model.predict(X) for model in self.models])
#         return np.mean(preds, axis=0)

# Main Training Model
def train_stacked_model(X_df, y_df, X_test_df, n_splits=5, seeds=[999]):
    final_preds_all = []
    oof_mapes = []

    for seed in seeds:
        print(f"\n Seed {seed}")
        final_preds = []

        for i, target in enumerate(y_df.columns):
            target_num = i + 1
            print(f"\n Modeling {target} ({target_num}/10)")

            y_target = y_df[target].values
            y_mean, y_std = y_target.mean(), y_target.std()
            y_norm = (y_target - y_mean) / y_std

            # SHAP feature selection
            lgb_selector = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                             force_col_wise=True, random_state=seed, verbose=-1)
            lgb_selector.fit(X_df, y_target)
            explainer = shap.Explainer(lgb_selector)
            shap_values = explainer(X_df)
            # 🔽 NEW: Save SHAP results
            shap_output = {
                "explainer": explainer,
                "shap_values": shap_values.values,
                "base_values": shap_values.base_values,
                "data": X_df.values,
                "feature_names": list(X_df.columns)
            }
            joblib.dump(shap_output, f"models/shap_target{target_num}.pkl")
            importances = np.abs(shap_values.values).mean(axis=0)
            indices = np.argsort(importances)[::-1]
            top_k = 30
            selected_features_names = X_df.columns[indices[:top_k]]

            X_target = X_df[selected_features_names].values
            X_test_target = X_test_df[selected_features_names].values

            oof_preds = np.zeros(X_target.shape[0])
            test_preds = np.zeros(X_test_target.shape[0])

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_target)):
                print(f"  Fold {fold+1}")
                X_tr, X_val = X_target[train_idx], X_target[val_idx]
                y_tr, y_val = y_norm[train_idx], y_norm[val_idx]

                # Base models
                lgb_model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.0125,
                                              force_col_wise=True, num_leaves=31, max_depth=8, verbose=-1, random_state=seed)
                xgb_model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.025, random_state=seed, objective='reg:squarederror')
                cat_model = CatBoostRegressor(iterations=2000, learning_rate=0.018, verbose=0, random_state=seed)
                nn_ensemble = NNEnsemble(n_models=3, random_state=seed)

                lgb_model.fit(X_tr, y_tr)
                xgb_model.fit(X_tr, y_tr)
                cat_model.fit(X_tr, y_tr)
                nn_ensemble.fit(X_tr, y_tr)

                val_preds_stack = np.vstack([
                    lgb_model.predict(X_val),
                    xgb_model.predict(X_val),
                    cat_model.predict(X_val),
                    nn_ensemble.predict(X_val)
                ]).T

                test_preds_fold = np.vstack([
                    lgb_model.predict(X_test_target),
                    xgb_model.predict(X_test_target),
                    cat_model.predict(X_test_target),
                    nn_ensemble.predict(X_test_target)
                ]).T

                meta_model = RidgeCV(alphas=np.logspace(-3, 2, 20), cv=5)
                meta_model.fit(val_preds_stack, y_val)

                oof_preds[val_idx] = meta_model.predict(val_preds_stack)
                test_preds += meta_model.predict(test_preds_fold) / n_splits

            # Denormalize
            oof_preds_final = oof_preds * y_std + y_mean
            test_preds_final = test_preds * y_std + y_mean

            # Isotonic calibration
            iso_reg = IsotonicRegression(y_min=y_target.min(), y_max=y_target.max(), out_of_bounds='clip')
            iso_reg.fit(oof_preds_final, y_target)

            oof_preds_calibrated = iso_reg.transform(oof_preds_final)
            test_preds_calibrated = iso_reg.transform(test_preds_final)

            # Blending
            best_mape = float('inf')
            best_alpha = 0.5
            blend_baseline = X_full[f'BlendWeighted_Property{target_num}'].values

            for alpha in np.linspace(0, 1, 101):
                blended = alpha * oof_preds_calibrated + (1 - alpha) * blend_baseline
                mape_try = mean_absolute_percentage_error(y_target, blended)
                if mape_try < best_mape:
                    best_mape = mape_try
                    best_alpha = alpha
            print(f"Best alpha for {target}: {best_alpha:.3f} | MAPE: {best_mape:.5f}")

            test_blend_baseline = X_test_full[f'BlendWeighted_Property{target_num}'].values
            post_blend = best_alpha * test_preds_calibrated + (1 - best_alpha) * test_blend_baseline
            post_blend = np.clip(post_blend, y_df[target].min(), y_df[target].max())

            oof_mapes.append(best_mape)
            final_preds.append(post_blend)

            # SAVE MODELS
            joblib.dump(scaler, "models/scaler.pkl")
            joblib.dump(X_full.columns.tolist(), "models/feature_columns.pkl")
            joblib.dump(lgb_model, f"models/lgb_target{target_num}.pkl")
            joblib.dump(xgb_model, f"models/xgb_target{target_num}.pkl")
            joblib.dump(cat_model, f"models/cat_target{target_num}.pkl")
            joblib.dump(nn_ensemble, f"models/nn_target{target_num}.pkl")
            joblib.dump(meta_model, f"models/meta_target{target_num}.pkl")
            joblib.dump(iso_reg, f"models/iso_target{target_num}.pkl")
            joblib.dump({
                "alpha": best_alpha,
                "mean": y_mean,
                "std": y_std,
                "min": y_target.min(),
                "max": y_target.max(),
                "selected_features": list(selected_features_names)
            }, f"models/blend_info_target{target_num}.pkl")

        final_preds_all.append(np.vstack(final_preds).T)

    final_preds_avg = np.mean(final_preds_all, axis=0)
    mean_mape = np.mean(oof_mapes)
    print(f"\nFinal CV MAPE across all targets: {mean_mape:.5f}")
    return final_preds_avg

# Train & Predict
submission_preds = train_stacked_model(X_df, y, X_test_df, n_splits=4)

# Save Submission
submission_df = pd.DataFrame({'ID': test_ids})
for i in range(10):
    submission_df[f"BlendProperty{i+1}"] = submission_preds[:, i]
submission_df.to_csv("output/submit_final.csv", index=False)
print("\nCalculating and saving property averages...")
property_averages = train[target_cols].mean().to_dict()
joblib.dump(property_averages, 'models/property_averages.pkl')
print("   - Property averages saved successfully.")