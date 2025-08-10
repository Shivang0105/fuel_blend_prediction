import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import uuid
import base64
import os
from streamlit_javascript import st_javascript
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Ba
# This section contains the real logic to load and run your models.
def generate_global_shap_summary(df, property_to_explain, assets):
    """
    Generates a global SHAP summary (beeswarm) plot for the entire dataset.
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
    # This is the computationally intensive step
    shap_values = explainer(X_full)

    # --- 4. Plotting ---
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8)) # Use a larger figure for this detailed plot

    # Generate the beeswarm summary plot
    shap.summary_plot(shap_values, X_full, show=False)
    
    fig = plt.gcf()
    ax = plt.gca()
    
    # Styling for dark theme
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    plt.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    
    # Find the colorbar axis and style its text
    try:
        cb_ax = fig.axes[1] 
        cb_ax.tick_params(labelcolor="white")
        cb_ax.set_ylabel(cb_ax.get_ylabel(), color="white")
    except IndexError:
        # Handle cases where a colorbar might not be present
        pass

    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)
def generate_shap_force_plot(row_data, property_to_explain, assets):
    """Generates a beautified SHAP force plot with rounded values for dark themes."""
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
    
    # --- THIS IS THE FIX ---
    # Create a new dataframe with values rounded to 3 decimal places for display
    X_display = X_single.copy()
    X_display.iloc[0] = X_display.iloc[0].round(2)

    # --- Plotting ---
    plt.style.use('dark_background')
    
    # Pass the rounded X_display data to the plot function
    fig = shap.force_plot(
        shap_values.base_values[0], 
        shap_values.values[0], 
        X_display.iloc[0], 
        matplotlib=True, 
        show=False
    )
    
    # Styling
    fig.patch.set_facecolor('#0E1117')
    for text in fig.findobj(plt.Text):
        text.set_color('white')

    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)
def generate_shap_decision_plot(row_data, property_to_explain, assets):
    """Generates a beautified SHAP decision plot for dark themes."""
    # --- Data prep is identical to the other functions ---
    target_num = int(property_to_explain.split('BlendProperty')[1])
    target_name = f'BlendProperty{target_num}'
    shap_assets_dict = assets['all_models'][target_name]['shap_explainer']
    explainer = shap_assets_dict['explainer'] 
    features_df = create_features(row_data)
    features_df = features_df.reindex(columns=assets['feature_columns'], fill_value=0)
    scaled_features = assets['scaler'].transform(features_df)
    X_single = pd.DataFrame(scaled_features, columns=assets['feature_columns'])
    shap_values = explainer(X_single)
    
    # --- Plotting ---
    plt.style.use('dark_background')

    # 1. Create a figure with a specific size first
    plt.figure(figsize=(10, 6))

    # 2. Call decision_plot WITHOUT the 'ax' argument. It will draw on the figure we just made.
    shap.decision_plot(shap_values.base_values[0], shap_values.values[0], X_single.iloc[0], show=False)
    
    # 3. Get the current figure and axes for styling
    fig = plt.gcf()
    ax = plt.gca()
    
    # Styling
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    plt.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    
    # Hide the top and right borders for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
                
                # --- ADD THIS LINE TO LOAD THE SHAP EXPLAINER ---
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
def generate_shap_waterfall_plot(row_data, property_to_explain, assets):
    """
    Generates a beautified SHAP waterfall plot with a custom size for dark themes.
    """
    # --- 1. SETUP & DATA PREPARATION (No changes here) ---
    target_num = int(property_to_explain.split('BlendProperty')[1])
    target_name = f'BlendProperty{target_num}'
    
    shap_assets_dict = assets['all_models'][target_name]['shap_explainer']
    explainer = shap_assets_dict['explainer'] 
    
    features_df = create_features(row_data)
    features_df = features_df.reindex(columns=assets['feature_columns'], fill_value=0)
    scaled_features = assets['scaler'].transform(features_df)
    
    X_single = pd.DataFrame(scaled_features, columns=assets['feature_columns'])
    shap_values = explainer(X_single)
    
    # --- 2. PLOTTING THE WATERFALL GRAPH ---
    N_FEATURES_TO_SHOW = 10
    plt.style.use('dark_background')
    
    # --- THIS IS THE LINE THAT CONTROLS THE SIZE ---
    # It creates the canvas that SHAP will draw on.
    plt.figure(figsize=(8, 6)) # Set a noticeable size (10-inch width, 6-inch height)

    # SHAP will now draw on the figure we just created and sized
    shap.waterfall_plot(shap_values[0], max_display=N_FEATURES_TO_SHOW, show=False)
    
    # Get the current figure and axes for further styling
    fig = plt.gcf()
    ax = plt.gca()
    
    # Styling for dark theme
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    plt.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    # Display the plot in Streamlit
    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)
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

    # all_predictions is a list of 10 arrays, each of shape (n_samples,).
    # Stack and transpose: shape (n_samples, 10)
    return np.vstack(all_predictions).T
import plotly.graph_objects as go

def plot_fraction_sums(df):
    """
    Creates a bar chart to visualize the sum of component fractions for each row,
    highlighting rows where the sum is not equal to 1.
    """
    temp_df = df.copy()
    # Dynamically find fraction columns to make it robust
    frac_cols = [col for col in temp_df.columns if 'fraction' in col and col.startswith('Component')]
    
    if not frac_cols:
        # Return an empty figure with a message if no fraction columns are found
        fig = go.Figure()
        fig.update_layout(title="No fraction columns found to validate.", paper_bgcolor='rgba(14, 17, 23, 1)', plot_bgcolor='rgba(14, 17, 23, 1)')
        return fig

    temp_df['fraction_sum'] = temp_df[frac_cols].sum(axis=1)
    temp_df['Status'] = np.where(np.isclose(temp_df['fraction_sum'], 1.0, atol=1e-4), 'Valid (Sum ≈ 1.0)', 'Invalid (Sum ≠ 1.0)')
    
    # Use 'ID' for the x-axis if it exists, otherwise use the dataframe index
    x_axis = temp_df['ID'] if 'ID' in temp_df.columns else temp_df.index

    fig = px.bar(
        temp_df,
        x=x_axis,
        y='fraction_sum',
        color='Status',
        title="⚖️ Fraction Sum Validation",
        labels={'fraction_sum': 'Sum of Fractions', 'x': 'Row ID'},
        color_discrete_map={
            'Valid (Sum ≈ 1.0)': '#28a745',  # Green
            'Invalid (Sum ≠ 1.0)': '#dc3545'  # Red
        },
        template="plotly_dark"
    )
    # Add a reference line at y=1.0 for easy comparison
    fig.add_hline(y=1.0, line_dash="dot", line_color="white", annotation_text="Target Sum = 1.0", annotation_position="bottom right")
    fig.update_layout(
        height=400,
        margin=dict(t=40, l=0, r=0, b=0),
        xaxis_title="Rows",
        yaxis_title="Sum of Fractions",
        paper_bgcolor='rgba(14, 17, 23, 1)',
        plot_bgcolor='rgba(14, 17, 23, 1)',
    )
    return fig
# Helper function to encode image to base64
def image_to_base64(path):
    """Converts a local image file to a base64 string."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
def plot_missing_matrix(df):
    import plotly.graph_objects as go
    import numpy as np

    mask = df.isnull().astype(int)

    fig = go.Figure(data=go.Heatmap(
        z=mask.values,
        x=list(range(mask.shape[1])),
        y=list(range(mask.shape[0])),
        colorscale=[[0, 'rgba(40, 167, 69, 0.7)'], [1, 'rgba(220, 53, 69, 1)']],
        zmin=0,  # Force the minimum of the color scale to 0
        zmax=1,  # Force the maximum of the color scale to 1
        showscale=False,
        hovertemplate='Row %{y}, Column %{x}<extra></extra>'
    ))

    fig.update_layout(
        title="🔍 Data Health Matrix",
        height=400,
        margin=dict(t=40, l=0, r=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, title="Features (Columns)"),
        yaxis=dict(showgrid=False, showticklabels=False, title="Rows"),
        paper_bgcolor='rgba(14, 17, 23, 1)',
        plot_bgcolor='rgba(14, 17, 23, 1)',
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
        steps = ["1. Upload Batch File", "2. Prediction Results","3: Sensitivity Analysis"]
    else:
        steps = []

    st.markdown("""
        <style>
            .step {
                text-align: center;
                padding: 0.5rem;
                border-bottom: 3px solid #ccc;
                flex-grow: 1;
            }
            .step.active {
                font-weight: 700;
                border-bottom: 3px solid #0072c6;
            }
            .step.completed {
                border-bottom: 3px solid #28a745;
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
    # Check if the dataframe is empty first
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
        # Find rows and columns with NaN values for a more descriptive error
        nan_locations = df.drop(columns=['ID'], errors='ignore').isnull()
        problem_rows = df.loc[nan_locations.any(axis=1), 'ID'].tolist()
        return False, f"Uploaded CSV contains missing (NaN) values. Check rows with IDs: {problem_rows[:5]}"

    frac_cols = [f"Component{i}_fraction" for i in range(1, num_components + 1)]
    
    # Check for negative fractions
    if (df[frac_cols] < 0).any().any():
        negative_rows = df.loc[(df[frac_cols] < 0).any(axis=1), 'ID'].tolist()
        return False, f"Component fractions cannot be negative. Check rows with IDs: {negative_rows[:5]}"

    # --- ✨ IMPROVED FRACTION SUM CHECK ✨ ---
    frac_sums = df[frac_cols].sum(axis=1)
    if not np.allclose(frac_sums, 1.0, atol=1e-4):
        bad_rows_indices = np.where(~np.isclose(frac_sums, 1.0, atol=1e-4))[0]
        
        error_messages = []
        for row_idx in bad_rows_indices[:5]: # Limit to the first 5 errors
            # Get the ID from the row, or use the index as a fallback
            row_id = df.iloc[row_idx].get('ID', f"index {row_idx}")
            actual_sum = frac_sums.iloc[row_idx]
            error_messages.append(f"row with ID '{row_id}' sums to {actual_sum:.4f}")
        
        full_error_string = "Component fractions must sum to 1.0. Found issues in: " + "; ".join(error_messages)
        return False, full_error_string
    # --- END OF IMPROVEMENT ---

    return True, "CSV is valid."
def get_contrasting_text_color(bg_color):
    """Returns black or white depending on the brightness of the background color"""
    bg_color = bg_color.lstrip('#')
    r, g, b = int(bg_color[0:2], 16), int(bg_color[2:4], 16), int(bg_color[4:6], 16)

    # Calculate luminance (ITU-R BT.709)
    brightness = (0.2126*r + 0.7152*g + 0.0722*b) / 255

    return '#000000' if brightness > 0.6 else '#FFFFFF'

def hex_to_rgba(hex_color, alpha):
    """Convert hex color to rgba string with specified alpha (opacity)"""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

def render_flow_block(title, subtitle, detail, color, icon="💡", width="300px"):
    block_id = f"flow-block-{uuid.uuid4().hex[:8]}"
    
    # Color Utilities
    def hex_to_rgba(hex_color, alpha=1.0):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {alpha})'

    background_rgba = hex_to_rgba(color, alpha=0.08)      # Translucent bg
    border_rgba = hex_to_rgba(color, alpha=0.5)           # Border stronger
    title_color = hex_to_rgba("#FFFFFF", 1.0)
    subtitle_color = hex_to_rgba("#FFFFFF", 0.75)
    detail_color = hex_to_rgba("#FFFFFF", 0.5)

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
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.3s ease-in-out;
        overflow: hidden;
        max-height: 160px;
        position: relative;
        backdrop-filter: blur(4px);
    }}

    .{block_id}:hover {{
        max-height: 320px;
        filter: brightness(1.05);
    }}

    .{block_id} .icon {{
        font-size: 1.8rem;
        margin-bottom: 8px;
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

def render_flow_diagram():
    gif_path = os.path.join("images", "arrow-down-navigation.gif")
    gif_base64 = get_gif_base64(gif_path)
    render_flow_block("Input Data","55 features","Contains 55 features per fuel blend: 5 volume fractions representing component percentages, and 50 component properties (10 per each of 5 components). These represent chemical, safety, and environmental attributes from real-world Certificates of Analysis (COA).","#6366F1","🗃️")
    st.markdown(f"""
        <div style='text-align: center; margin-top: -12px; margin-bottom: -12px;'>
            <img src="data:image/gif;base64,{gif_base64}" width="60" />
        </div>
    """, unsafe_allow_html=True)
    render_flow_block("Feature Engineering","BlendWeighted Features","Generates new features by calculating blend-weighted averages of properties, residuals between component and blend values, and statistical summaries to enhance input data. This transforms raw data into more informative features for better model learning.","#6B7280","🛠️")
    st.markdown(f"""
        <div style='text-align: center; margin-top: -12px; margin-bottom: -12px;'>
            <img src="data:image/gif;base64,{gif_base64}" width="60" />
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])   
    with col1:
        render_flow_block("LightGBM","Base Model","A high-performance, open-source gradient-boosting framework developed by Microsoft. It uses leaf-wise tree growth and histogram-based algorithms to optimize training speed and memory while maintaining high accuracy, especially for large datasets.","#10B981","🌲",260)
    with col2:
        render_flow_block("XGBoost","Base Model","An efficient, scalable gradient-boosting algorithm known for speed and accuracy. It builds trees level-wise, includes regularization to prevent overfitting, and supports parallel processing. Widely popular for structured data and competition-winning models.","#EF9F44","🚀",260)
    with col3:
        render_flow_block("CatBoost","Base Model","A gradient boosting method designed to handle categorical and numerical data natively without preprocessing. Uses symmetric trees and ordered target encoding for accuracy and speed, reducing overfitting and preprocessing effort, suitable for diverse datasets.","#F59E0B","🐱",260)
    with col4:
        render_flow_block("Neural Net","55 features","A machine learning model inspired by the human brain, consisting of interconnected nodes (neurons) organized in layers. It learns complex patterns from data through weighted connections, enabling tasks like classification, regression, and pattern recognition.","#EC4899","🧠",260)
    st.markdown(f"""
        <div style='text-align: center; margin-top: -12px; margin-bottom: -12px;'>
            <img src="data:image/gif;base64,{gif_base64}" width="60" />
        </div>
    """, unsafe_allow_html=True)
    render_flow_block("Meta Model","RidgeCV Ensemble","The Meta Model uses RidgeCV to linearly combine base model predictions, finding the best regularization to reduce overfitting. It learns optimal weights to stack outputs into a single, robust final prediction for each property, improving accuracy and generalization.","#DC2626","🧰")
    st.markdown(f"""
        <div style='text-align: center; margin-top: -12px; margin-bottom: -12px;'>
            <img src="data:image/gif;base64,{gif_base64}" width="60" />
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])   
    with col1:
        render_flow_block("Calibration","Isotonic Regression & Probability Calibration","Calibration uses Isotonic Regression to adjust ensemble predictions, reducing bias and aligning outputs closer to observed data. This non-parametric method improves reliability and accuracy, ensuring predictions are realistic and generalize well to new data.","#38BDF8","📈",480)
    with col2:
        render_flow_block("Final Output","10 Blend Properties with Optimized Prediction","The Final Output combines calibrated predictions with baseline weighted averages to produce accurate estimates for 10 blend properties per sample. This optimized blending reduces errors, delivering reliable and actionable results for fuel blend optimization.","#14B8A6","🎯",480)
def render_online_status():
    st.markdown('''
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
        <div style="width:10px;height:10px;background:#22c55e;border-radius:50%;animation:pulse 2s infinite ease-in-out;"></div>
        <div style="color:#a3f0c1;font-size:0.9em;">
            <strong style="color:#22c55e;">Online</strong> – System Ready
        </div>
    </div>
    <style>
    @keyframes pulse {
        0%, 100% {transform: scale(1); opacity: 1;}
        50% {transform: scale(1.6); opacity: 0.5;}
    }
    </style>
    ''', unsafe_allow_html=True)

# Wrap run_sensitivity_analysis with arguments
def worker(args):
    prop, row_data, assets, component_to_vary = args
    return prop, run_sensitivity_analysis(row_data, assets, prop, component_to_vary)


# ✅ Lottie loading function
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()       
# --- 3. Main Application ---
def main():
    # --- This is the ONLY place set_page_config should be called ---
    st.set_page_config(page_title="Fuel Blend AI", layout="wide")
    st_javascript("window.scrollTo(0, 0);")
    # --- Load assets at the top so they are available to all steps ---
    assets = load_assets()
    if assets is None:
        st.stop() # Stop the app if models can't be loaded

    # Initialize session state for multi-step navigation
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if st.session_state.step == 0:
        # Global page styling and animation CSS
        st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
                html, body, [class*="css"] {
                    font-family: 'Inter', sans-serif;
                    background-color: #0d1117;
                    color: #cbd5e1;
                }
                .stApp { background-color: #0d1117; }

                /* Gate: hide rest until title animation completes */
                .hidden-until-ready {
                    opacity: 0;
                    pointer-events: none;
                    transition: opacity 0.6s ease;
                }
                .hidden-until-ready.ready {
                    opacity: 1;
                    pointer-events: auto;
                }

                /* Title animation with glow */
                @keyframes zoomAndSettle {
                    0% {
                        transform: scale(2.0);
                        opacity: 0;
                        text-shadow: none;
                    }
                    50% {
                        transform: scale(2.0);
                        opacity: 1;
                        text-shadow: 0 0 30px rgba(212,175,55,0.5), 0 0 60px rgba(248,241,229,0.3);
                    }
                    100% {
                        transform: scale(1);
                        opacity: 1;
                        text-shadow: 0 0 8px rgba(212,175,55,0.2);
                    }
                }


                /* Class-driven trigger so we can force animation to start after mount */
                #text-logo {
                    opacity: 0;
                    transform: scale(1);
                    background: linear-gradient(135deg, #d4af37, #f8f1e5, #c0a060);
                    -webkit-background-clip: text;
                    background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 4.5em;
                    font-weight: 900;
                    letter-spacing: 1px;
                    line-height: 1.2;
                    margin: 0.5em 0;
                    will-change: transform, opacity, text-shadow;
                }

                #text-logo.animate {
                    animation: zoomAndSettle 2.8s forwards cubic-bezier(0.16, 1, 0.3, 1) !important;
                }

                /* Sub-elements fade in after title */
                @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

                .hero-container {
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    padding: 2rem;
                    overflow: hidden;
                }

                .hero-subtitle, .hero-tagline, .scroll-button-wrapper {
                    opacity: 0;
                    animation: fadeIn 1.5s forwards;
                }
                .hero-subtitle { animation-delay: 2.8s; font-size: 1.25em; color: #94a3b8; }
                .hero-tagline { animation-delay: 3.1s; font-size: 1.4em; font-weight: 700; margin-top: 1.5em; }
                .hero-tagline .highlight { color: #4ade80; }
                .scroll-button-wrapper { animation-delay: 3.4s; margin-top: 2.5em; }

                #scroll-button {
                    background: transparent; border: 2px solid #4ade80; color: #4ade80;
                    padding: 10px 22px; font-size: 1em; font-weight: 700;
                    border-radius: 50px; cursor: pointer; transition: all 0.3s ease;
                }
                #scroll-button:hover { background-color: #4ade80; color: #0d1117; }

                .section-header {
                    text-align: center; font-size: 2.2em; font-weight: 700;
                    margin-top: 2em; margin-bottom: 1em; color: #f1f5f9;
                }
                #text-logo {
                    opacity: 0;
                    transform: scale(1);
                    background: linear-gradient(135deg, #d4af37, #f8f1e5, #c0a060);
                    -webkit-background-clip: text;
                    background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 4.5em;
                    font-weight: 900;
                    letter-spacing: 1px;
                    line-height: 1.2;
                    margin: 0.5em 0;
                    will-change: transform, opacity, text-shadow;
                }
                #text-logo.animate {
                    animation: zoomAndSettle 2.8s forwards cubic-bezier(0.16, 1, 0.3, 1) !important;
                }
                    
            </style>
        """, unsafe_allow_html=True)

        # Hero section: shows first, animates title
        st.markdown("""
            <div class="hero-container">
                <h1 id="text-logo">Shell.ai Fuel Blend Challenge</h1>
                <p class="hero-subtitle">Machine Learning-Powered Fuel Blend Property Prediction</p>
                <p class="hero-tagline">In a world chasing <span class="highlight">net-zero</span>, fuel is no longer just a commodity — it's a <span class="highlight">climate lever.</span></p>
                <div class="scroll-button-wrapper">
                    <button id="scroll-button">See How It Works ↓</button>
                </div>
            </div>
        """, unsafe_allow_html=True)
        # Force-start the title animation via class toggle (robust to Streamlit remounts)
        st.markdown("""
            <script>
                const title = document.getElementById('text-logo');
                if (title) {
                    title.classList.remove('animate');
                    void title.offsetWidth; // reflow to reset animation
                    title.classList.add('animate');
                }
                // Reset scroll on first paint to avoid racing with Streamlit layout
                window.requestAnimationFrame(() => window.scrollTo(0, 0));
            </script>
        """, unsafe_allow_html=True)

        # Gate wrapper: rest of the content stays hidden until animation end
        st.markdown('<div id="main-content" class="hidden-until-ready">', unsafe_allow_html=True)

        # ----- Your existing sectioned content below -----
        lottie_url = "https://assets9.lottiefiles.com/packages/lf20_vgiqdeca.json"
        lottie_json = load_lottieurl(lottie_url)
        if lottie_json:
            st_lottie(lottie_json, height=300, speed=1, quality="high")

        st.markdown('<div class="section-header">How We Solve It</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            render_flow_block(
                "Calibrated Predictions",
                "Confidence-tuned ensemble outputs.",
                "Our models are calibrated to provide not just predictions, but a reliable measure of confidence, ensuring trustworthy results.",
                "#2ECC71", "📈", "95%"
            )
        with col2:
            render_flow_block(
                "Feature Engineering",
                "Creates derived features & weights.",
                "Automated creation of hundreds of insightful features that capture complex interactions between components.",
                "#4B4BAF", "🧮", "95%"
            )
        with col3:
            render_flow_block(
                "Model Stacking",
                "Combines strengths of multiple learners.",
                "We use a meta-learning approach, where a final model learns to optimally weigh the predictions from our base models.",
                "#7F8C8D", "🛠️", "95%"
            )

        st.markdown('<div class="section-header">What Powers Our Predictions</div>', unsafe_allow_html=True)
        render_flow_diagram()

        st.markdown("""
            <div style="text-align:center; padding:2em 0; margin-top:2em;">
                <h2 style="color:#4ade80;">Are You Ready to Predict the Future of Fuel?</h2>
                <p style="color:#94a3b8;">Step inside the AI-powered lab that helps design sustainable fuel blends at scale.</p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Launch Prediction Tool", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

        # Close gate wrapper
        st.markdown('</div>', unsafe_allow_html=True)

        # JS: reveal main content after title animation completes; enable smooth scroll
        st.markdown("""
            <script>
                const gate = document.getElementById('main-content');
                const scrollButton = document.getElementById('scroll-button');
                const title2 = document.getElementById('text-logo');

                if (title2 && gate) {
                    // Reveal when the animation ends
                    title2.addEventListener('animationend', (e) => {
                        if (e.animationName === 'zoomAndSettle') {
                            gate.classList.add('ready');
                        }
                    });
                }

                if (scrollButton && gate) {
                    scrollButton.onclick = function() {
                        gate.scrollIntoView({ behavior: 'smooth' });
                    }
                }
            </script>
        """, unsafe_allow_html=True)

        return

# --- ✨ Display the progress bar on all subsequent steps ---
    display_step_progress(st.session_state.step, mode="batch")
    # STEP 1: Upload CSV
    if st.session_state.step == 1:
        st.header("Step 1: Upload Batch File")

        # --- File Upload Logic (no changes here) ---
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])
            # --- ADD THIS SNIPPET ---
            with st.expander("❓ Not sure about the file format?"):
                    st.info(
                        "The CSV should contain an 'ID' column, 5 'ComponentX_fraction' columns, "
                        "and 50 'ComponentX_PropertyY' columns. The component fractions for each row must sum to 1.0."
                    )
                    st.markdown("Click the **Load Example Data** button to see a working example.")
            # --- END SNIPPET ---
        with col2:
            st.markdown("</br>", unsafe_allow_html=True)
            if st.button("Load Example Data", use_container_width=True):
                try:
                    example_df = pd.read_csv("datasets/test.csv").head(10)
                    st.session_state.batch_input_df = example_df
                    st.rerun()
                except FileNotFoundError:
                    st.error("Could not find 'datasets/test.csv'.")

        df_to_process = None # Use a new variable name for clarity
        if uploaded_file:
            df_to_process = pd.read_csv(uploaded_file)
        elif "batch_input_df" in st.session_state:
            df_to_process = st.session_state.batch_input_df

        if df_to_process is not None:
            
            st.info("📝 Review your data below. The graphs will check for issues.")
            edited_df = st.data_editor(df_to_process, use_container_width=True, num_rows="dynamic", key="editable_csv")

            # --- ✨ NEW: Workaround for the data editor bug ---
            # If the editor returns an empty table, fall back to the original data.
            if edited_df.empty:
                st.warning("⚠️ The data editor returned an empty table. Reverting to the original data.", icon="🤖")
                final_df = df_to_process
            else:
                final_df = edited_df

            st.session_state.batch_input_df = final_df

            # --- File Health Analysis Section ---
            st.subheader("🕵️ File Health Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_missing_matrix(final_df), use_container_width=True)
                st.caption("💡 **COACH'S TIP:** This matrix shows where data is missing (red spots). "
                            "A fully green chart means your data is complete and healthy!")
            with col2:
                st.plotly_chart(plot_fraction_sums(final_df), use_container_width=True)
                st.caption("⚖️ **VALIDATION:** This chart checks if your fractions add up to 1.0. "
                            "Any red bars indicate rows that need to be fixed in the editor above.")
            
            st.markdown("---")

            # --- Final validation now uses the corrected dataframe and function ---
            is_valid, msg = validate_batch_input(final_df)
            if not is_valid:
                st.error(f"❌ **Validation Failed:** {msg}")
            else:
                st.success("✅ **Validation Passed:** Your data looks good! You can now proceed to prediction.")
            col1,col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Introduction", use_container_width=True, disabled=not is_valid):
                    st.session_state.step = 0
                    st.rerun()
            with col2:
                if st.button("➡️ Predict", use_container_width=True, disabled=not is_valid):
                    st.session_state.step = 2
                    st.rerun()
    # STEP 2: Predict
    elif st.session_state.step == 2:
        st.header("Step 2: Prediction Results")
        
        if "batch_input_df" not in st.session_state:
            st.warning("⚠️ No batch file uploaded.")
            if st.button("⬅️ Back to Upload"):
                st.session_state.step = 1
                st.rerun()
        else:
            df = st.session_state.batch_input_df
            with st.spinner("🔄 Running batch predictions..."):
                try:
                    predictions = predict_properties(df, assets)
                    pred_df = pd.DataFrame(predictions, columns=[f"BlendProperty{i}" for i in range(1, 11)])
                    # This is the full DataFrame with all columns
                    final_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

                    st.session_state.final_prediction_df = final_df

                    st.subheader("📊 Prediction Results")
                    
                    # --- This part creates the filtered view for the screen ---
                    cols_to_display = ["ID"] + [f"BlendProperty{i}" for i in range(1, 11)]
                    st.dataframe(final_df[cols_to_display].style.format("{:.4f}"), use_container_width=True)

                    st.markdown("---")
                    
                    # --- ✨ NEW: Column Selection for Download ✨ ---
                    st.subheader("⬇️ Download Custom CSV")
                    
                    # Get the list of all available columns from the full results
                    all_columns = final_df.columns.tolist()
                    
                    # Define the default columns to select (ID + predictions)
                    default_columns = ["ID"] + [f"BlendProperty{i}" for i in range(1, 11)]
                    
                    # Create a multiselect widget for the user to choose columns
                    selected_columns = st.multiselect(
                        "Select columns to include in your download:",
                        options=all_columns,
                        default=all_columns # Default to a clean selection
                    )

                    if selected_columns:
                        # Filter the DataFrame based on the user's column selection
                        df_to_download = final_df[selected_columns]
                        
                        # Convert the selected data to CSV
                        csv_selected = df_to_download.to_csv(index=False).encode('utf-8')
                        
                        # Create a download button specifically for the selected columns
                        st.download_button(
                            label=f"Download Selected ({len(selected_columns)} columns)",
                            data=csv_selected,
                            file_name="custom_blend_predictions.csv",
                            mime="text/csv",
                            key="download_custom"
                        )
                    else:
                        st.info("Select one or more columns to enable the download button.")
                    
                    # --- End of New Section ---

                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("⬅️ Upload Batch File", use_container_width=True):
                            for key in ["batch_input_df", "final_prediction_df"]:
                                st.session_state.pop(key, None)
                            st.session_state.step = 1
                            st.rerun()
                    with col2:
                        if st.button("➡️ Analysis",use_container_width=True):
                            st.session_state.step = 3
                            st.rerun()
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")


    # STEP 3: Row-Level Analysis
    elif st.session_state.step == 3:
        st.header("Step 3: Blend Analysis & Explainability")
        
        # Check if prediction data exists, otherwise send user back
        if "final_prediction_df" not in st.session_state:
            st.warning("⚠️ No prediction data available. Please go back to Step 2.")
            if st.button("⬅️ Back to Prediction Results"):
                st.session_state.step = 2
                st.rerun()
            return
            
        df = st.session_state.final_prediction_df

        # --- Section 1: Overall Dataset Analysis ---
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
                title="📦 Distribution of Component Fractions Across All Uploaded Blends"
            )
            fig_box.update_layout(
                height=400, paper_bgcolor='rgba(14, 17, 23, 1)',
                plot_bgcolor='rgba(14, 17, 23, 1)', font=dict(color="#EAEAEA"),
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("---")

        # --- Section 2: Single Row Deep Dive ---
        st.markdown(f"<h2>🔬 Single Blend Deep Dive</h2>", unsafe_allow_html=True)
        st.markdown("Select a single blend from your data to inspect its composition, understand its prediction, and run 'what-if' scenarios.")

        selected_id = st.selectbox("Select a Blend ID to analyze:", df["ID"].unique())
        row_data = df[df["ID"] == selected_id]
        
        st.subheader("📋 Selected Row Composition")
        st.dataframe(row_data, use_container_width=True, height=80)
        # Radar Chart for Component Fractions
        col1, col2 = st.columns(2)

        # Shared component list
        components = [f"Component{i}_fraction" for i in range(1, 6)]

        # First radar: Original fractions
        with col1:
            fractions = [row_data.iloc[0][comp] for comp in components]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=fractions + fractions[:1],
                theta=components + [components[0]],
                mode='lines',
                fill='toself',
                name='Original Fractions',
                line=dict(color="#3498db"),
                fillcolor='rgba(52, 152, 219, 0.4)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='#444654'),
                    angularaxis=dict(tickfont=dict(size=12), rotation=90),
                    bgcolor='rgba(14, 17, 23, 1)'
                ),
                paper_bgcolor='rgba(14, 17, 23, 1)',
                plot_bgcolor='rgba(14, 17, 23, 1)',
                font=dict(color="#EAEAEA"),
                showlegend=False,
                height=400,
                title=dict(text="Original Component Fractions")
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        with col2:
            properties = [f"BlendProperty{i}" for i in range(1, 11)]
            fractions_modified = [row_data.iloc[0][prop] for prop in properties]

            # Calculate the range dynamically
            max_val = max(fractions_modified)
            radial_range = [0, max_val + .1]

            fig_radar_modified = go.Figure()
            fig_radar_modified.add_trace(go.Scatterpolar(
                r=fractions_modified + fractions_modified[:1],
                theta=properties + [properties[0]],
                mode='lines',
                fill='toself',
                name='Modified Fractions',
                line=dict(color="#e67e22"),
                fillcolor='rgba(230, 126, 34, 0.4)'
            ))
            fig_radar_modified.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=radial_range, gridcolor='#444654'),
                    angularaxis=dict(tickfont=dict(size=12), rotation=90),
                    bgcolor='rgba(14, 17, 23, 1)'
                ),
                paper_bgcolor='rgba(14, 17, 23, 1)',
                plot_bgcolor='rgba(14, 17, 23, 1)',
                font=dict(color="#EAEAEA"),
                showlegend=False,
                height=400,
                title=dict(text="Modified Component Fractions")
            )
            st.plotly_chart(fig_radar_modified, use_container_width=True)



        # --- Replace your existing SHAP section with this ---
        st.markdown("<h3>💡 Prediction Explanation (Why?)</h3>", unsafe_allow_html=True)
        with st.expander("How does this work?"):
            st.info(
            "This section explains *why* the model made its prediction. Choose a property and a plot type to see "
            "which features had the biggest impact."
            )
        # Create columns for the selectors
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

        # Call the correct function based on user's choice
        if property_to_explain:
            with st.spinner(f"Generating {plot_type} for {property_to_explain}..."):
                if plot_type == "Waterfall":
                    generate_shap_waterfall_plot(row_data, property_to_explain, assets)
                elif plot_type == "Decision Plot":
                    generate_shap_decision_plot(row_data, property_to_explain, assets)
                elif plot_type == "Force Plot":
                    generate_shap_force_plot(row_data, property_to_explain, assets)
        # --- End of updated SHAP section ---

        # --- Sensitivity Analysis Section ---
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
            progress_bar = st.progress(10, text="🚀 Launching parallel prediction threads...")
            
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
                yaxis_title="Predicted Value", template="plotly_dark", height=600
            )
            st.plotly_chart(fig_sensitivity, use_container_width=True)
        if st.button("⬅️ Prediction", use_container_width=True):
            st.session_state.step = 0
            st.rerun()

                
if __name__ == "__main__":
    main()