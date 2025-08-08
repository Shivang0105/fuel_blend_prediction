# test_predict.py

import pandas as pd
from predict_blend_properties import load_all_models_all_targets, predict_single_row

# Load a single test row
row = pd.read_csv("datasets/test.csv").iloc[[0]]
row = row.drop(columns="ID", errors="ignore")

# Load all models once
models_dict = load_all_models_all_targets()

# Predict all 10 blend properties
preds = predict_single_row(row, models_dict)
print("Predicted Blend Properties:", preds)
