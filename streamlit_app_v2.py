import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import uuid
import base64
from streamlit_javascript import st_javascript
from streamlit_lottie import st_lottie
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time
import shap
import matplotlib.pyplot as plt

# --- 1. Backend & Logic Functions ---
# This section contains the real logic to load and run your models.
def generate_global_shap_summary(df, property_to_explain, assets):
    """
    Generates a global SHAP summary (beeswarm) plot for the entire dataset, styled for a light theme.
    """
    # --- 1. Setup ---
    target_num = int(property_to_explain.split('BlendProperty')[1])
    target_name = f'BlendProperty{target_num}'
    shap_assets_dict = assets['all_models'][target_name]['shap_explainer']
    explainer = shap_assets_dict['explainer']

    # --- 2. Preprocess the ENTIRE dataframe ---
    features_df = create_features(df)
    features_df = features_df.reindex(columns=assets['feature_columns'], fill_value=0)
    scaled_features = assets['scaler'].transform(features_df)
    X_full = pd.DataFrame(scaled_features, columns=assets['feature_columns'])

    # --- 3. Calculate SHAP values for the ENTIRE dataframe ---
    shap_values = explainer(X_full)

    # --- 4. Plotting (Light Theme) ---
    plt.figure(figsize=(12, 8))

    # Generate the beeswarm summary plot
    shap.summary_plot(shap_values, X_full, show=False)

    fig = plt.gcf()
    ax = plt.gca()

    # Styling for light theme
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.tick_params(colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')

    # Find the colorbar axis and style its text
    try:
        cb_ax = fig.axes[1]
        cb_ax.tick_params(labelcolor="black")
        cb_ax.set_ylabel(cb_ax.get_ylabel(), color="black")
    except IndexError:
        pass

    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)

def generate_shap_force_plot(row_data, property_to_explain, assets):
    """Generates a SHAP force plot with rounded values for light themes, showing only original features."""
    # --- Data prep ---
    target_num = int(property_to_explain.split('BlendProperty')[1])
    target_name = f'BlendProperty{target_num}'
    shap_assets_dict = assets['all_models'][target_name]['shap_explainer']
    explainer = shap_assets_dict['explainer']
    features_df = create_features(row_data)
    features_df = features_df.reindex(columns=assets['feature_columns'], fill_value=0)
    scaled_features = assets['scaler'].transform(features_df)
    X_single = pd.DataFrame(scaled_features, columns=assets['feature_columns'])
    shap_values = explainer(X_single)

    # --- MODIFIED: Filter features to show only original inputs ---
    filtered_shap_values, filtered_X_single = _filter_shap_features(shap_values, X_single)

    X_display = filtered_X_single.copy()
    X_display.iloc[0] = X_display.iloc[0].round(2)

    # --- Plotting ---
    fig = shap.force_plot(
        filtered_shap_values.base_values[0],
        filtered_shap_values.values[0],
        X_display.iloc[0],
        matplotlib=True,
        show=False,
        text_rotation=10
    )

    # Styling for light theme
    fig.patch.set_facecolor('white')
    for text in fig.findobj(plt.Text):
        text.set_color('black')

    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)

def generate_shap_decision_plot(row_data, property_to_explain, assets):
    """Generates a beautified SHAP decision plot for light themes, showing only original features."""
    # --- Data prep ---
    target_num = int(property_to_explain.split('BlendProperty')[1])
    target_name = f'BlendProperty{target_num}'
    shap_assets_dict = assets['all_models'][target_name]['shap_explainer']
    explainer = shap_assets_dict['explainer']
    features_df = create_features(row_data)
    features_df = features_df.reindex(columns=assets['feature_columns'], fill_value=0)
    scaled_features = assets['scaler'].transform(features_df)
    X_single = pd.DataFrame(scaled_features, columns=assets['feature_columns'])
    shap_values = explainer(X_single)

    # --- MODIFIED: Filter features to show only original inputs ---
    filtered_shap_values, filtered_X_single = _filter_shap_features(shap_values, X_single)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    shap.decision_plot(
        filtered_shap_values.base_values[0],
        filtered_shap_values.values[0],
        filtered_X_single.iloc[0],
        show=False
    )

    fig = plt.gcf()
    ax = plt.gca()

    # Styling for light theme
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.tick_params(colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)

def generate_shap_waterfall_plot(row_data, property_to_explain, assets):
    """
    Generates a SHAP waterfall plot for light themes, showing only original features.
    """
    # --- 1. SETUP & DATA PREPARATION ---
    target_num = int(property_to_explain.split('BlendProperty')[1])
    target_name = f'BlendProperty{target_num}'
    shap_assets_dict = assets['all_models'][target_name]['shap_explainer']
    explainer = shap_assets_dict['explainer']
    features_df = create_features(row_data)
    features_df = features_df.reindex(columns=assets['feature_columns'], fill_value=0)
    scaled_features = assets['scaler'].transform(features_df)
    X_single = pd.DataFrame(scaled_features, columns=assets['feature_columns'])
    shap_values = explainer(X_single)

    # --- MODIFIED: Filter features to show only original inputs ---
    filtered_shap_values, _ = _filter_shap_features(shap_values, X_single)

    # --- 2. PLOTTING THE WATERFALL GRAPH ---
    N_FEATURES_TO_SHOW = 20
    plt.figure(figsize=(8, 6))

    # Pass the first instance of the filtered explanation object
    shap.waterfall_plot(filtered_shap_values[0], max_display=N_FEATURES_TO_SHOW, show=False)

    fig = plt.gcf()
    ax = plt.gca()

    # Styling for light theme
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.tick_params(colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')

    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)

def create_features(df):
    """This function must be IDENTICAL to the one in your training script."""
    df = df.copy()
    for i in range(1, 11):
        df[f'BlendWeighted_Property{i}'] = sum(df[f'Component{j}_fraction'] * df[f'Component{j}_Property{i}'] for j in range(1, 6))
    for i in range(1, 11):
        blend = df[f'BlendWeighted_Property{i}']
        for j in range(1, 6):
            df[f'Residual_Component{j}_Prop{i}'] = df[f'Component{j}_Property{i}'] - blend
    for j in range(1, 6):
        props = [f'Component{j}_Property{i}' for i in range(1, 11)]
        df[f'Component{j}_mean'] = df[props].mean(axis=1)
        df[f'Component{j}_std'] = df[props].std(axis=1)
    for j in range(1, 6):
        for i in range(1, 11):
            df[f'Frac{j}_x_Prop{i}'] = df[f'Component{j}_fraction'] * df[f'Component{j}_Property{i}']
    for i in range(1, 11):
        props = [f"Component{j}_Property{i}" for j in range(1, 6)]
        df[f'Property{i}_max'] = df[props].max(axis=1)
        df[f'Property{i}_min'] = df[props].min(axis=1)
        df[f'Property{i}_std'] = df[props].std(axis=1)
    frac_cols = [f"Component{i}_fraction" for i in range(1, 6)]
    df["frac_sum"] = df[frac_cols].sum(axis=1)
    df["frac_max"] = df[frac_cols].max(axis=1)
    df["frac_min"] = df[frac_cols].min(axis=1)
    df["frac_std"] = df[frac_cols].std(axis=1)
    return df

@st.cache_resource
def load_assets():
    """Loads all pre-trained assets, including SHAP explainers, from the 'models' directory."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")

        all_models = {}
        for i in range(1, 11):
            target_num = i
            all_models[f'BlendProperty{target_num}'] = {
                'lgbm': joblib.load(os.path.join(models_dir, f'lgb_target{target_num}.pkl')),
                'xgb': joblib.load(os.path.join(models_dir, f'xgb_target{target_num}.pkl')),
                'cat': joblib.load(os.path.join(models_dir, f'cat_target{target_num}.pkl')),
                'nn_ensemble': joblib.load(os.path.join(models_dir, f'nn_target{target_num}.pkl')),
                'meta_model': joblib.load(os.path.join(models_dir, f'meta_target{target_num}.pkl')),
                'iso_reg': joblib.load(os.path.join(models_dir, f'iso_target{target_num}.pkl')),
                'shap_explainer': joblib.load(os.path.join(models_dir, f'shap_target{target_num}.pkl')),
                'blend_info': joblib.load(os.path.join(models_dir, f'blend_info_target{target_num}.pkl'))
            }

        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        feature_columns = scaler.get_feature_names_out()

        return {
            "all_models": all_models,
            "scaler": scaler,
            "feature_columns": feature_columns,
        }
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please ensure all model and SHAP files are in the 'models' directory.")
        return None

def predict_properties(input_df, assets):
    """
    Runs the prediction pipeline for single or batch DataFrame input.
    Always returns a numpy array of shape (n_samples, 10).
    """
    features_df = create_features(input_df)
    features_df = features_df.reindex(columns=assets['feature_columns'], fill_value=0)
    scaled_features = assets['scaler'].transform(features_df)
    scaled_df = pd.DataFrame(scaled_features, columns=assets['feature_columns'])

    all_predictions = []

    for i in range(1, 11):
        target = f"BlendProperty{i}"
        models = assets['all_models'][target]
        blend_info = models['blend_info']
        target_features = blend_info['selected_features']
        X = scaled_df[target_features].values

        base_preds = np.vstack([
            models['lgbm'].predict(X),
            models['xgb'].predict(X),
            models['cat'].predict(X),
            models['nn_ensemble'].predict(X)
        ]).T  # shape (n_samples, 4)

        meta_pred_norm = models['meta_model'].predict(base_preds)
        meta_pred = meta_pred_norm * blend_info['std'] + blend_info['mean']
        calibrated_pred = models['iso_reg'].transform(meta_pred)
        baseline = features_df[f'BlendWeighted_Property{i}'].values
        alpha = blend_info['alpha']
        final_blend = alpha * calibrated_pred + (1 - alpha) * baseline
        final_prediction = np.clip(final_blend, blend_info['min'], blend_info['max'])
        all_predictions.append(final_prediction)  # shape (n_samples,)

    return np.vstack(all_predictions).T

def plot_fraction_sums(df):
    """
    Creates a bar chart to visualize the sum of component fractions for each row,
    highlighting rows where the sum is not equal to 1.
    """
    temp_df = df.copy()
    frac_cols = [col for col in temp_df.columns if 'fraction' in col and col.startswith('Component')]

    if not frac_cols:
        fig = go.Figure()
        fig.update_layout(title="No fraction columns found to validate.")
        return fig

    temp_df['fraction_sum'] = temp_df[frac_cols].sum(axis=1)
    temp_df['Status'] = np.where(np.isclose(temp_df['fraction_sum'], 1.0, atol=1e-4), 'Valid (Sum ≈ 1.0)', 'Invalid (Sum ≠ 1.0)')

    x_axis = temp_df['ID'] if 'ID' in temp_df.columns else temp_df.index

    fig = px.bar(
        temp_df,
        x=x_axis,
        y='fraction_sum',
        color='Status',
        title="⚖ Fraction Sum Validation",
        labels={'fraction_sum': 'Sum of Fractions', 'x': 'Row ID'},
        color_discrete_map={
            'Valid (Sum ≈ 1.0)': '#28a745',  # Green
            'Invalid (Sum ≠ 1.0)': '#dc3545'   # Red
        },
        template="plotly_white"
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Target Sum = 1.0", annotation_position="bottom right")
    fig.update_layout(
        height=400,
        margin=dict(t=40, l=0, r=0, b=0),
        xaxis_title="Rows",
        yaxis_title="Sum of Fractions",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig
def image_to_base64(path):
    """Converts a local image file to a base64 string."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
    
def plot_missing_matrix(df):
    mask = df.isnull().astype(int)

    fig = go.Figure(data=go.Heatmap(
        z=mask.values,
        x=list(range(mask.shape[1])),
        y=list(range(mask.shape[0])),
        colorscale=[[0, 'rgba(40, 167, 69, 0.7)'], [1, 'rgba(220, 53, 69, 1)']],
        zmin=0,
        zmax=1,
        showscale=False,
        hovertemplate='Row %{y}, Column %{x}<extra></extra>'
    ))

    fig.update_layout(
        title="🔍 Data Health Matrix",
        height=400,
        margin=dict(t=40, l=0, r=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, title="Features (Columns)"),
        yaxis=dict(showgrid=False, showticklabels=False, title="Rows"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240, 242, 246, 0.8)', # A slight off-white for the plot area
    )
    return fig



def run_sensitivity_analysis(input_df, assets, property_to_analyze, component_to_vary):
    base_fractions = [input_df[f'Component{i}_fraction'].iloc[0] for i in range(1, 6)]
    analysis_results = []
    for new_fraction in np.linspace(0, 1, 20):
        temp_df = input_df.copy()
        vary_idx = component_to_vary - 1
        temp_df[f'Component{component_to_vary}_fraction'] = new_fraction
        other_indices = [i for i in range(5) if i != vary_idx]
        sum_of_others_base = sum(base_fractions[i] for i in other_indices)

        if sum_of_others_base > 0:
            remaining_fraction = 1.0 - new_fraction
            for i in other_indices:
                proportion = base_fractions[i] / sum_of_others_base
                temp_df[f'Component{i+1}_fraction'] = remaining_fraction * proportion

        predictions = predict_properties(temp_df, assets)
        property_index = int(property_to_analyze.split('BlendProperty')[1]) - 1
        analysis_results.append({
            'varied_fraction': new_fraction,
            'predicted_value': float(predictions[0, property_index])
        })
    return pd.DataFrame(analysis_results)

# --- 2. UI Component Functions ---
def display_step_progress(step, mode):
    if mode == "single":
        steps = ["1. Composition Fractions", "2. Component Properties", "3. Prediction Results"]
    elif mode == "batch":
        steps = ["1. Upload Batch File", "2. Prediction Results","3: Blend Analysis"]
    else:
        steps = []

    st.markdown("""
        <style>
            .step {
                text-align: center;
                padding: 0.5rem;
                border-bottom: 3px solid #ccc;
                flex-grow: 1;
                color: #4F4F4F;
            }
            .step.active {
                font-weight: 700;
                border-bottom: 3px solid #0072c6; /* Blue for active */
                color: #0072c6;
            }
            .step.completed {
                border-bottom: 3px solid #28a745; /* Green for completed */
            }
        </style>
    """, unsafe_allow_html=True)

    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps)):
        with col:
            css_class = "step"
            if i + 1 == step:
                css_class += " active"
            elif i + 1 < step:
                css_class += " completed"
            st.markdown(f'<div class="{css_class}">{step_name}</div>', unsafe_allow_html=True)

    st.markdown("---")
def validate_batch_input(df, num_components=5, num_properties=10):
    """
    Validates uploaded batch CSV with more descriptive error messages.
    """
    if df.empty:
        return False, "The data table is empty. Please ensure data is loaded and not deleted."

    required_cols = ["ID"] + \
        [f"Component{i}_fraction" for i in range(1, num_components + 1)] + \
        [f"Component{i}_Property{j}" for i in range(1, num_components + 1) for j in range(1, num_properties + 1)]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Uploaded CSV is missing required columns: {missing_cols}"

    extra_cols = [col for col in df.columns if col not in required_cols]
    if extra_cols:
        return False, f"Uploaded CSV has unexpected extra columns: {extra_cols}"

    if df.drop(columns=['ID'], errors='ignore').isnull().any().any():
        nan_locations = df.drop(columns=['ID'], errors='ignore').isnull()
        problem_rows = df.loc[nan_locations.any(axis=1), 'ID'].tolist()
        return False, f"Uploaded CSV contains missing (NaN) values. Check rows with IDs: {problem_rows[:5]}"

    frac_cols = [f"Component{i}_fraction" for i in range(1, num_components + 1)]

    if (df[frac_cols] < 0).any().any():
        negative_rows = df.loc[(df[frac_cols] < 0).any(axis=1), 'ID'].tolist()
        return False, f"Component fractions cannot be negative. Check rows with IDs: {negative_rows[:5]}"

    frac_sums = df[frac_cols].sum(axis=1)
    if not np.allclose(frac_sums, 1.0, atol=1e-4):
        bad_rows_indices = np.where(~np.isclose(frac_sums, 1.0, atol=1e-4))[0]

        error_messages = []
        for row_idx in bad_rows_indices[:5]:
            row_id = df.iloc[row_idx].get('ID', f"index {row_idx}")
            actual_sum = frac_sums.iloc[row_idx]
            error_messages.append(f"row with ID '{row_id}' sums to {actual_sum:.4f}")

        full_error_string = "Component fractions must sum to 1.0. Found issues in: " + "; ".join(error_messages)
        return False, full_error_string

    return True, "CSV is valid."

def render_flow_block(title, subtitle, detail, color, icon="💡", width="300px"):
    block_id = f"flow-block-{uuid.uuid4().hex[:8]}"

    def hex_to_rgba(hex_color, alpha=1.0):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {alpha})'

    background_rgba = hex_to_rgba(color, alpha=0.10)
    border_rgba = hex_to_rgba(color, alpha=0.6)
    # --- UPDATED TEXT COLORS TO BLUE ---
    title_color = "#005A9C"      # Dark Blue
    subtitle_color = "#0072c6"   # Standard Blue
    detail_color = "#4DA8DA"     # Light Blue

    st.markdown(f"""
    <div class="{block_id}">
        <div class="icon">{icon}</div>
        <div class="title">{title}</div>
        <div class="subtitle">{subtitle}</div>
        <div class="detail">{detail}</div>
    </div>

    <style>
    .{block_id} {{
        background: {background_rgba};
        border: 1.5px solid {border_rgba};
        border-radius: 12px;
        padding: 12px;
        width: {width};
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
        margin: 12px auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease-in-out;
        overflow: hidden;
        max-height: 160px;
        position: relative;
    }}

    .{block_id}:hover {{
        max-height: 320px;
        border-color: {color};
        box-shadow: 0 6px 15px rgba(0,0,0,0.12);
    }}

    .{block_id} .icon {{
        font-size: 1.8rem;
        margin-bottom: 8px;
        color: {color};
    }}

    .{block_id} .title {{
        font-size: 1rem;
        font-weight: 700;
        color: {title_color};
        margin-bottom: 4px;
    }}

    .{block_id} .subtitle {{
        font-size: 0.85rem;
        color: {subtitle_color};
        line-height: 1.2;
        margin-bottom: 8px;
    }}

    .{block_id} .detail {{
        font-size: 0.80rem;
        color: {detail_color};
        line-height: 1.4;
        opacity: 0;
        max-height: 0;
        transition: opacity 0.3s ease, max-height 0.3s ease;
    }}

    .{block_id}:hover .detail {{
        opacity: 1;
        max-height: 500px;
    }}
    </style>
    """, unsafe_allow_html=True)
def get_gif_base64(gif_path):
    with open(gif_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _filter_shap_features(shap_explanation, features_df):
    """
    Filters a SHAP Explanation object and a feature DataFrame to include only
    the original input features (ComponentX_fraction and ComponentX_PropertyY).
    """
    # Get all feature names from the DataFrame
    all_feature_names = features_df.columns.tolist()

    # Define the pattern for the features we want to keep
    desired_features = [
        f for f in all_feature_names
        if 'Component' in f and ('_fraction' in f or '_Property' in f)
    ]

    # If no desired features are found, return the originals to avoid errors
    if not desired_features:
        return shap_explanation, features_df

    # Get indices from the SHAP explanation's feature list to slice its arrays correctly
    try:
        shap_feature_names = shap_explanation.feature_names
        desired_indices = [shap_feature_names.index(f) for f in desired_features]
    except (ValueError, AttributeError):
        # Fallback if a feature name is missing or the attribute doesn't exist
        return shap_explanation, features_df

    # Filter the SHAP Explanation object's internal arrays
    filtered_values = shap_explanation.values[:, desired_indices]
    filtered_data = shap_explanation.data[:, desired_indices]

    # Create the new, filtered Explanation object
    new_explanation = shap.Explanation(
        values=filtered_values,
        base_values=shap_explanation.base_values,
        data=filtered_data,
        feature_names=desired_features
    )

    # Filter the corresponding DataFrame
    filtered_features_df = features_df[desired_features]

    return new_explanation, filtered_features_df

def render_flow_diagram():
    gif_path = os.path.join("images", "arrow-down-navigation.gif")
    gif_base64 = get_gif_base64(gif_path)
    render_flow_block("Input Data","55 features per blend","5 volume fractions and 50 component properties from real-world Certificates of Analysis (COA), defining chemical, safety, and environmental attributes.","#6366F1","🗃")
    st.markdown(f"<div style='text-align: center; margin-top: -12px; margin-bottom: -12px;'><img src='data:image/gif;base64,{gif_base64}' width='60' style='filter: invert(1);' /></div>", unsafe_allow_html=True)
    render_flow_block("Feature Engineering","Creates blend-weighted features","Generates weighted averages, residuals, and statistical summaries to transform raw data into more informative features for better model learning.","#6B7280","🛠")
    st.markdown(f"<div style='text-align: center; margin-top: -12px; margin-bottom: -12px;'><img src='data:image/gif;base64,{gif_base64}' width='60' style='filter: invert(1);' /></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        render_flow_block("LightGBM","Base Model","A high-performance gradient-boosting framework using leaf-wise tree growth for fast, memory-efficient, and accurate training.","#10B981","🌲",260)
    with col2:
        render_flow_block("XGBoost","Base Model","An efficient, scalable gradient-boosting algorithm known for speed and accuracy, with built-in regularization to prevent overfitting.","#3B82F6","🚀",260)
    with col3:
        render_flow_block("CatBoost","Base Model","A gradient boosting method that natively handles categorical data using symmetric trees, reducing overfitting and preprocessing effort.","#F59E0B","🐱",260)
    with col4:
        render_flow_block("Neural Net","Base Model","A model inspired by the human brain, consisting of layered nodes that learn complex patterns and non-linear relationships from data.","#EC4899","🧠",260)
    st.markdown(f"<div style='text-align: center; margin-top: -12px; margin-bottom: -12px;'><img src='data:image/gif;base64,{gif_base64}' width='60' style='filter: invert(1);' /></div>", unsafe_allow_html=True)
    render_flow_block("Meta Model","RidgeCV Ensemble","Linearly combines base model predictions, using RidgeCV to find the best regularization strength and learn optimal weights for a robust final prediction.","#DC2626","🧰")
    st.markdown(f"<div style='text-align: center; margin-top: -12px; margin-bottom: -12px;'><img src='data:image/gif;base64,{gif_base64}' width='60' style='filter: invert(1);' /></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        render_flow_block("Calibration","Isotonic Regression","Adjusts ensemble predictions using non-parametric Isotonic Regression, reducing systematic bias and aligning outputs closer to observed data for improved reliability.","#38BDF8","📈",480)
    with col2:
        render_flow_block("Final Output","10 Optimized Predictions","Combines calibrated predictions with baseline weighted averages to produce accurate, reliable, and actionable estimates for 10 key blend properties.","#14B8A6","🎯",480)

def worker(args):
    prop, row_data, assets, component_to_vary = args
    return prop, run_sensitivity_analysis(row_data, assets, prop, component_to_vary)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- 3. Main Application ---
def main():
    st.set_page_config(page_title="Fuel Blend AI", layout="wide")
    st_javascript("window.scrollTo(0, 0);")

    # --- ✨ NEW: Patched Gradient Background Theme ---
    st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
        }
                
        /* Main app background with vibrant gradient patches on top & strong white fade on bottom center */
        .stApp {
            background-color: #FFFFFF; /* Pure white base */
            background-image:
                /* Color patches are kept in the top half */
                radial-gradient(at 10% 20%, hsla(340, 100%, 75%, 0.85) 0px, transparent 50%),
                radial-gradient(at 90% 10%, hsla(200, 100%, 75%, 0.85) 0px, transparent 50%),
                radial-gradient(at 75% 30%, hsla(50, 100%, 75%, 0.75) 0px, transparent 50%),
                radial-gradient(at 25% 35%, hsla(210, 100%, 80%, 0.75) 0px, transparent 50%),

                /* White fade for bottom half */
                linear-gradient(to bottom, rgba(255,255,255,0) 40%, rgba(255,255,255,1) 60%);
                
            background-repeat: no-repeat;
            background-attachment: fixed;
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            color: #1A1A1A; /* Darker neutral text */
        }

        /* Transparent header & blurred sidebar */
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
        }

        /* Popovers and menus stay solid white */
        div[data-baseweb="popover"], div[data-baseweb="menu"] > ul {
            background-color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)

    assets = load_assets()
    if assets is None:
        st.stop()

    if 'step' not in st.session_state:
        st.session_state.step = 0

    # --- Step 0: Landing Page ---
    if st.session_state.step == 0:
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }
            .big-title {
                font-size: 3.8em;
                font-weight: 900;
                text-align: center;
                background: linear-gradient(to right, #0072c6 20%, #28a745 50%, #0072c6 80%);
                background-size: 200% auto;
                color: #000;
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: shine 4s linear infinite;
            }
            @keyframes shine {
                to {
                    background-position: 200% center;
                }
            }
                    
            .subtitle { font-size: 1.25em; color: #005A9C; text-align: center; margin-bottom: 2em; }
            .section-header { text-align: center; font-size: 2.2em; font-weight: 700; margin-top: 2em; margin-bottom: 1em; color: #005A9C; }

            /* --- BRIGHTER SUBTLE MATTE BLUE → CYAN BUTTON --- */
            div[data-testid="stButton"] > button {
                font-size: 1.3em !important; /* force size */
                font-weight: 800 !important; /* force bold */
                padding: 0.9em 1.2em;
                border-radius: 50px;
                background-image: linear-gradient(45deg, #0057b7 30%, #00d4ff 70%);
                color: white !important;
                border: none;
                transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
                box-shadow: 0 4px 18px rgba(0, 212, 255, 0.35);
                text-shadow: 1px 1px 2px rgba(0,0,0,0.15);
            }

            /* make sure the text inside also scales */
            div[data-testid="stButton"] > button > div > p,
            div[data-testid="stButton"] > button > span {
                font-size: 1.3em !important;
                font-weight: 800 !important;
            }

            div[data-testid="stButton"] > button:hover {
                transform: scale(1.05);
                box-shadow: 0 8px 28px rgba(0, 212, 255, 0.45);
            }



            .logo-container {
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 1rem 0;
                margin-bottom: -1rem;
                perspective: 800px;
            }
            #interactive-logo-img {
                max-width: 250px;
                cursor: pointer;
                filter: none;
                transform: scale(1);
                transition: transform 0.4s ease-out, filter 0.4s ease-out;
            }
            .logo-container:hover #interactive-logo-img {
                filter: drop-shadow(-8px 0 6px rgba(59, 130, 246, 0.7))
                        drop-shadow(8px 0 6px rgba(74, 222, 128, 0.7));
                transform: scale(1.1);
                transition: transform 0.05s linear, filter 0.4s ease-out;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # --- Page Content ---
        with st.container():
            try:
                logo_path = "images/unnamed-removebg-preview.png"
                logo_base64 = image_to_base64(logo_path)
                _ , col2, _ = st.columns([1, 1, 1])
                with col2:
                    st.markdown(f"""
                    <div class="logo-container">
                        <img id="interactive-logo-img" src="data:image/png;base64,{logo_base64}">
                    </div>
                    """, unsafe_allow_html=True)
            except FileNotFoundError:
                _ , col2, _ = st.columns([1, 1, 1])
                with col2:
                    st.error("⚠ Logo image not found.")

            st.markdown('<div class="big-title">Shell.ai Fuel Blend Hackathon</div>', unsafe_allow_html=True)
            st.markdown('<div class="subtitle">Reimagining sustainable energy with AI-powered fuel property prediction.</div>', unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align:center; max-width:850px; margin:auto; padding-top: 1.5em;">
                <p style="font-size: 1.35em; font-weight: 700; color: #005A9C; margin-bottom: 0.6em;">
                    In a world chasing <span style="color:#28a745;">net-zero</span>, fuel is no longer just a commodity — it's a <span style="color:#0072c6;">climate lever</span>.
                </p>
            </div>
            """, unsafe_allow_html=True)

            lottie_url = "https://assets9.lottiefiles.com/packages/lf20_vgiqdeca.json"
            lottie_json = load_lottieurl(lottie_url)
            if lottie_json:
                st_lottie(lottie_json, height=280, speed=1, quality="high")

            # --- Team Section ---
            st.markdown("""
            <div style="
                margin: 2em auto; 
                max-width: 600px; 
                padding: 2em 3em; 
                border-radius: 15px;
                background: linear-gradient(135deg, rgba(135,206,250,0.3), rgba(255,182,193,0.3));
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 182, 193, 0.3);
                text-align: center;
                color: #005A9C;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
                <h3 style="font-weight: 700; font-size: 2.2em; margin-bottom: 0.5em; color: #0072c6;">Team Locus</h3>
                <div style="display: flex; justify-content: center; gap: 6em; font-size: 1.5em; font-weight: 600;">
                    <div style="text-align: left;">
                        <p style="margin: 0.3em 0;">Abhinav Tyagi</p>
                        <p style="margin: 0.3em 0;">Siddharth Bansal</p>
                    </div>
                    <div style="text-align: left;">
                        <p style="margin: 0.3em 0;">Shivang Sharma</p>
                        <p style="margin: 0.3em 0;">Utkarsh Singh</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align:center; padding:2em 0; margin-top:2em;">
                <h2 style="color:#0072c6;">Are You Ready to Predict the Future of Fuel?</h2>
                <p style="color:#005A9C;">Step inside the AI-powered lab that helps design sustainable fuel blends at scale.</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Launch Prediction Tool", use_container_width=True):
                st.session_state.step = 1
                st.rerun()

            st.markdown('<div class="section-header">How We Solve It</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                render_flow_block("Calibrated Predictions", "Confidence-tuned ensemble outputs.", "Our models are calibrated to provide not just predictions, but a reliable measure of confidence, ensuring trustworthy results.", "#2ECC71", "📈",200)
            with col2:
                render_flow_block("Feature Engineering", "Creates derived features & weights.", "Automated creation of hundreds of insightful features that capture complex interactions between components.", "#3B82F6", "🧮",200)
            with col3:
                render_flow_block("Model Stacking", "Combines strengths of multiple learners.", "We use a meta-learning approach, where a final model learns to optimally weigh the predictions from our base models.", "#6B7280", "🛠",200)

            st.markdown('<div class="section-header">What Powers Our Predictions</div>', unsafe_allow_html=True)
            render_flow_diagram()

        return
    
    with st.container():
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button("🏠 Home"):
                st.session_state.step = 0
                for key in ["batch_input_df", "final_prediction_df"]:
                    st.session_state.pop(key, None)
                st.rerun()

    display_step_progress(st.session_state.step, mode="batch")
    # STEP 1: Upload CSV
    if st.session_state.step == 1:
        st.header("Step 1: Upload Batch File")

        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])
            st.markdown(
            """
            *Your CSV file must have:*
            - An ID column.
            - *5* ComponentX_fraction columns (X in 1-5).
            - *50* ComponentX_PropertyY columns(X in 1-5 and Y in 1-10).
            - The component fractions for each row must sum to *1.0*.
            - 56 columns in total.

            Click 'Load Example Data' to see a working example.
            """
            )

        with col2:
            # Remove <br>, add margin-top for vertical alignment
            st.markdown("<div style='margin-top: 40px;'>", unsafe_allow_html=True)
            if st.button("Load Example Data", use_container_width=True):
                try:
                    example_df = pd.read_csv("datasets/test.csv").head(10)
                    st.session_state.batch_input_df = example_df
                    st.rerun()
                except FileNotFoundError:
                    st.error("Could not find 'datasets/test.csv'.")
            st.markdown("</div>", unsafe_allow_html=True)

        df_to_process = None
        if uploaded_file:
            df_to_process = pd.read_csv(uploaded_file)
        elif "batch_input_df" in st.session_state:
            df_to_process = st.session_state.batch_input_df

        if df_to_process is not None:
            st.info("Review your data below. The graphs will check for issues.")
            edited_df = st.data_editor(df_to_process, use_container_width=True, num_rows="dynamic", key="editable_csv")

            if edited_df.empty:
                st.warning("⚠ The data editor returned an empty table. Reverting to the original data.", icon="🤖")
                final_df = df_to_process
            else:
                final_df = edited_df

            st.session_state.batch_input_df = final_df

            st.subheader("File Health Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_missing_matrix(final_df), use_container_width=True)
                st.markdown(
                    '<p style="color:black; text-align: center;"><b>COACH\'S TIP:</b> This matrix shows missing data (blue spots). A fully light-gray chart is healthy!</p>',
                    unsafe_allow_html=True
                )
            with col2:
                st.plotly_chart(plot_fraction_sums(final_df), use_container_width=True)
                st.markdown(
                    '<p style="color:black; text-align: center;"><b>VALIDATION:</b> This chart checks if fractions sum to 1.0. Red bars show rows needing fixes.</p>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

            is_valid, msg = validate_batch_input(final_df)
            if not is_valid:
                st.error(f" Validation Failed :( : {msg}")
            else:
                st.success("Validation Passed :) : Your data looks good! You can now proceed to prediction.")

            if st.button("➡ Predict", use_container_width=True, disabled=not is_valid):
                st.session_state.step = 2
                st.rerun()
    # STEP 2: Predict
    elif st.session_state.step == 2:
        st.header("Step 2: Prediction Results")

        if "batch_input_df" not in st.session_state:
            st.warning("⚠ No batch file uploaded.")
            if st.button("⬅ Back to Upload"):
                st.session_state.step = 1
                st.rerun()
        else:
            df = st.session_state.batch_input_df
            with st.spinner("🔄 Running batch predictions..."):
                try:
                    predictions = predict_properties(df, assets)
                    pred_df = pd.DataFrame(predictions, columns=[f"BlendProperty{i}" for i in range(1, 11)])
                    final_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
                    st.session_state.final_prediction_df = final_df

                    st.subheader("📊 Prediction Results")

                    cols_to_display = ["ID"] + [f"BlendProperty{i}" for i in range(1, 11)]
                    st.dataframe(final_df[cols_to_display].style.format("{:.4f}"), use_container_width=True)
                    st.markdown("---")

                    # --- SIMPLIFIED DOWNLOAD BUTTON ---
                    csv_to_download = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                       label="📥 Download Full Results CSV",
                       data=csv_to_download,
                       file_name="blend_predictions_output.csv",
                       mime="text/csv",
                       use_container_width=True,
                       key="download_full_csv"
                    )
                    st.markdown("---")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("⬅ Upload Another File", use_container_width=True):
                            for key in ["batch_input_df", "final_prediction_df"]:
                                st.session_state.pop(key, None)
                            st.session_state.step = 1
                            st.rerun()
                    with col2:
                        if st.button("➡ Go to Analysis",use_container_width=True):
                            st.session_state.step = 3
                            st.rerun()
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

    # STEP 3: Blend-Level Analysis
    elif st.session_state.step == 3:
        st.header("Step 3: Blend Analysis & Explainability")

        if "final_prediction_df" not in st.session_state:
            st.warning("⚠ No prediction data available. Please go back to Step 2.")
            if st.button("⬅ Back to Prediction Results"):
                st.session_state.step = 2
                st.rerun()
            return

        df = st.session_state.final_prediction_df

        st.markdown(f"<h2>📊 Overall Dataset Analysis</h2>", unsafe_allow_html=True)
        st.markdown(
            "This shows how component fractions are distributed across the entire uploaded batch, "
            "helping you spot overall trends and distributions."
        )

        numeric_df = df.select_dtypes(include='number')
        fraction_cols = [col for col in numeric_df.columns if 'fraction' in col]

        if fraction_cols:
            melted_frac = numeric_df[fraction_cols].melt(var_name="Component", value_name="Fraction")
            fig_box = px.box(
                melted_frac, x="Component", y="Fraction", points="all", color="Component",
                title="📦 Distribution of Component Fractions Across All Uploaded Blends",
                template="plotly_white"
            )
            fig_box.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")

        st.markdown(f"<h2>🔬 Single Blend Deep Dive</h2>", unsafe_allow_html=True)
        st.markdown("Select a single blend from your data to inspect its composition, understand its prediction, and run 'what-if' scenarios.")

        selected_id = st.selectbox("Select a Blend ID to analyze:", df["ID"].unique())
        row_data = df[df["ID"] == selected_id]

        st.subheader("📋 Selected Row Composition")
        st.dataframe(row_data, use_container_width=True, height=80)

        # --- Side-By-Side Radar Charts with Alignment ---
        st.subheader("📡 Blend Composition Radars")

        # --- CONTROLS MOVED ABOVE COLUMNS FOR ALIGNMENT ---
        comp_to_show = 1

        col1, col2 = st.columns(2)
        with col1:
            # Radar Chart 1: Component Fractions
            components = [f"Component {i}" for i in range(1, 6)]
            frac_cols = [f"Component{i}_fraction" for i in range(1, 6)]
            fractions = [row_data.iloc[0][comp] for comp in frac_cols]

            fig_radar_frac = go.Figure()
            fig_radar_frac.add_trace(go.Scatterpolar(
                r=fractions + fractions[:1],
                theta=components + [components[0]],
                mode='lines', fill='toself', name='Fractions',
                line=dict(color="#0072c6"), # Blue
                fillcolor='rgba(0, 114, 198, 0.4)'
            ))
            fig_radar_frac.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='#DDDDDD'),
                    angularaxis=dict(tickfont=dict(size=12), rotation=90),
                    bgcolor='rgba(255, 255, 255, 0.5)'
                ),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#000000"), showlegend=False, height=400,
                title=dict(text="Component Fractions")
            )
            st.plotly_chart(fig_radar_frac, use_container_width=True)

        with col2:
            # Radar Chart 2: Component Properties
            prop_labels = [f"Prop {i}" for i in range(1, 11)]
            prop_cols = [f"Component{comp_to_show}_Property{i}" for i in range(1, 11)]
            prop_values = row_data.iloc[0][prop_cols].values.tolist()

            fig_radar_props = go.Figure()
            fig_radar_props.add_trace(go.Scatterpolar(
                r=prop_values + prop_values[:1],
                theta=prop_labels + [prop_labels[0]],
                mode='lines', fill='toself', name='Properties',
                line=dict(color="#28a745"), # Green
                fillcolor='rgba(40, 167, 69, 0.4)'
            ))
            fig_radar_props.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, gridcolor='#DDDDDD'),
                    angularaxis=dict(tickfont=dict(size=12)),
                    bgcolor='rgba(255, 255, 255, 0.5)'
                ),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#000000"), showlegend=False, height=400,
                title=dict(text=f"Properties for Component {comp_to_show}")
            )
            st.plotly_chart(fig_radar_props, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True) # Add some space
        # --- End of radar section ---

        st.markdown("<h3>💡 Prediction Explanation (Why?)</h3>", unsafe_allow_html=True)
        with st.expander("How does this work?"):
            st.info(
            "This section explains why the model made its prediction. Choose a property and a plot type to see "
            "which features had the biggest impact."
            )
        col1, col2 = st.columns(2)
        with col1:
            property_to_explain = st.selectbox(
                "Select a Blend Property to explain:",
                [f"BlendProperty{i}" for i in range(1, 11)],
                key="shap_property_selector"
            )
        with col2:
            plot_type = st.selectbox(
                "Select a plot type:",
                [ "Force Plot","Waterfall", "Decision Plot"],
                key="shap_plot_selector"
            )

        if property_to_explain:
            with st.spinner(f"Generating {plot_type} for {property_to_explain}..."):
                if plot_type == "Waterfall":
                    generate_shap_waterfall_plot(row_data, property_to_explain, assets)
                elif plot_type == "Decision Plot":
                    generate_shap_decision_plot(row_data, property_to_explain, assets)
                elif plot_type == "Force Plot":
                    generate_shap_force_plot(row_data, property_to_explain, assets)

        st.markdown("<h3>🔬 Sensitivity Analysis (What If?)</h3>", unsafe_allow_html=True)
        with st.expander("How does this work?"):
            st.info(
                """
                This tool helps you play 'what-if'. Select a component to vary its fraction from 0% to 100%.
                The model then re-calculates all 10 blend properties at each step, showing you how sensitive they are to changes in that single component.
                """
            )

        component_to_vary = st.selectbox("Select Component to Vary", [1, 2, 3, 4, 5], format_func=lambda x: f"Component {x}")

        if st.button("Run Sensitivity Analysis", use_container_width=True):
            blend_props = [f"BlendProperty{i}" for i in range(1, 11)]
            tasks = [(prop, row_data.copy(), assets, component_to_vary) for prop in blend_props]
            progress_bar = st.progress(0, text="🚀 Launching parallel prediction threads...")

            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(worker, args): i for i, args in enumerate(tasks)}
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    prop, analysis_df = future.result()
                    results.append((prop, analysis_df))
                    progress_bar.progress((i + 1) / len(tasks), text=f"✅ Completed {prop}...")

            progress_bar.empty()

            fig_sensitivity = go.Figure()
            for prop, analysis_df in sorted(results, key=lambda x: int(x[0].replace("BlendProperty", ""))):
                fig_sensitivity.add_trace(go.Scatter(
                    x=analysis_df['varied_fraction'],
                    y=analysis_df['predicted_value'].astype(float),
                    mode='lines+markers', name=prop
                ))

            fig_sensitivity.update_layout(
                title=f"Sensitivity Analysis: Varying Component {component_to_vary}",
                xaxis_title=f"Fraction of Component {component_to_vary}",
                yaxis_title="Predicted Value", template="plotly_white", height=600,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_sensitivity, use_container_width=True)

if __name__ == "__main__":
    main()