# feature_engineering.py

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
