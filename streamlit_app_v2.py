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
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
import shap
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder
from scipy.optimize import minimize, differential_evolution
import pandas as pd
import plotly.graph_objects as go
from streamlit.components.v1 import html
from streamlit_js_eval import streamlit_js_eval
import io  

# --- 1. Backend & Logic Functions ---
def generate_global_shap_summary(df, property_to_explain, assets):
    """
    Generates a global SHAP summary (beeswarm) plot for a SAMPLE of the dataset, 
    styled for a light theme, to prevent memory errors.
    """
    # --- 1. Setup ---
    target_num = int(property_to_explain.split('BlendProperty')[1])
    target_name = f'BlendProperty{target_num}'
    shap_assets_dict = assets['all_models'][target_name]['shap_explainer']
    explainer = shap_assets_dict['explainer']

    # --- 2. Preprocess the dataframe (with subsampling) ---
    
    # --- SOLUTION: Subsample the data if it's too large ---
    SAMPLE_SIZE = 200 
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        df_sample = df
    # --- End of change ---

    # Process the sample, not the full dataframe
    features_df = create_features(df_sample)
    features_df = features_df.reindex(columns=assets['feature_columns'], fill_value=0)
    scaled_features = assets['scaler'].transform(features_df)
    X_sample = pd.DataFrame(scaled_features, columns=assets['feature_columns'])

    # --- 3. Calculate SHAP values for the SAMPLE ---
    shap_values = explainer(X_sample)
    
    # Key Step: Call the helper function to filter features before plotting
    filtered_shap_values, filtered_X_single = _filter_shap_features(shap_values, X_sample)

    # --- 4. Plotting (Light Theme) ---
    # Restored a reasonable figure size
    plt.figure(figsize=(1, 1))

    # Generate the beeswarm summary plot using the FILTERED data
    shap.summary_plot(filtered_shap_values, filtered_X_single, show=False)

    fig = plt.gcf()
    ax = plt.gca()

    # Styling for light theme
    fig.patch.set_facecolor('white')
    fig.set_figwidth(6) # Set width
    fig.set_figheight(4) 
    ax.set_facecolor('white')
    plt.tick_params(colors='black',axis='y', labelsize=4)
    plt.tick_params(axis='x', labelsize=4)
    ax.xaxis.label.set_size(4)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')
    # Find the colorbar axis and style its text
    try:
        cb_ax = fig.axes[1]
        cb_ax.tick_params(labelsize=4, labelcolor="black")
        cb_ax.set_ylabel(cb_ax.get_ylabel(), fontsize=8, color="black")
    except IndexError:
        pass
    
    # Added a title to the plot for context
    plt.title(f"Global Feature Importance for {property_to_explain}", fontsize=8, color='black')

    st.pyplot(fig, bbox_inches='tight', use_container_width=False)
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
    """
    Generates a SHAP decision plot and displays it as a static image
    to ensure correct styling in Streamlit.
    """
    # --- 1. Data Prep ---
    target_num = int(property_to_explain.split('BlendProperty')[1])
    target_name = f'BlendProperty{target_num}'
    shap_assets_dict = assets['all_models'][target_name]['shap_explainer']
    explainer = shap_assets_dict['explainer']
    features_df = create_features(row_data)
    features_df = features_df.reindex(columns=assets['feature_columns'], fill_value=0)
    scaled_features = assets['scaler'].transform(features_df)
    X_single = pd.DataFrame(scaled_features, columns=assets['feature_columns'])
    shap_values = explainer(X_single)

    filtered_shap_values, filtered_X_single = _filter_shap_features(shap_values, X_single)

    # --- 2. Plotting ---
    try:
        # Set the default font size. This will be 'baked' into the final image.
        plt.rcParams['font.size'] = 4 # Adjust for your desired size

        # Create the SHAP plot in the background
        shap.decision_plot(
            filtered_shap_values.base_values[0],
            filtered_shap_values.values[0],
            filtered_X_single.iloc[0],
            show=False
        )

        fig = plt.gcf()
        ax = plt.gca()

        # Apply all your custom styling for the light theme
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

        # --- KEY CHANGES: Render plot to an image buffer ---
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches='tight')
        
        # Display the static image from the buffer
        st.image(buf)

    finally:
        # IMPORTANT: Reset Matplotlib's settings to default
        plt.rcdefaults()
        
    # Close the figure object to free up memory
    plt.close(fig)

def generate_shap_waterfall_plot(row_data, property_to_explain, assets):
    """
    Generates a SHAP waterfall plot and displays it as a static image
    to ensure correct font sizes in Streamlit.
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

    filtered_shap_values, _ = _filter_shap_features(shap_values, X_single)
    
    N_FEATURES_TO_SHOW = 20

    # --- 2. PLOTTING ---
    try:
        # Set the default font size. This will be 'baked' into the final image.
        plt.rcParams['font.size'] = 4  # Adjust for your desired size

        shap.waterfall_plot(
            filtered_shap_values[0], 
            max_display=N_FEATURES_TO_SHOW, 
            show=False
        )

        fig = plt.gcf()
        fig.patch.set_facecolor('white')

        # --- KEY CHANGES: Render plot to an image buffer ---
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches='tight')
        
        # Display the static image from the buffer
        st.image(buf)

    finally:
        # IMPORTANT: Reset Matplotlib's settings to default
        plt.rcdefaults()
        
    # Close the figure object to free up memory
    plt.close(fig)

def create_features(df):
    """This function is an optimized version that avoids fragmentation."""
    df_original = df.copy()
    new_cols = {} # Use a dictionary to store new columns

    # Calculate BlendWeighted properties
    for i in range(1, 11):
        col_name = f'BlendWeighted_Property{i}'
        new_cols[col_name] = sum(df_original[f'Component{j}_fraction'] * df_original[f'Component{j}_Property{i}'] for j in range(1, 6))

    # Calculate Residuals
    for i in range(1, 11):
        blend = new_cols[f'BlendWeighted_Property{i}']
        for j in range(1, 6):
            col_name = f'Residual_Component{j}_Prop{i}'
            new_cols[col_name] = df_original[f'Component{j}_Property{i}'] - blend

    # Calculate Component stats
    for j in range(1, 6):
        props = [f'Component{j}_Property{i}' for i in range(1, 11)]
        new_cols[f'Component{j}_mean'] = df_original[props].mean(axis=1)
        new_cols[f'Component{j}_std'] = df_original[props].std(axis=1)

    # Calculate Fraction x Property interactions
    for j in range(1, 6):
        for i in range(1, 11):
            col_name = f'Frac{j}_x_Prop{i}'
            new_cols[col_name] = df_original[f'Component{j}_fraction'] * df_original[f'Component{j}_Property{i}']

    # Calculate Property stats
    for i in range(1, 11):
        props = [f"Component{j}_Property{i}" for j in range(1, 6)]
        new_cols[f'Property{i}_max'] = df_original[props].max(axis=1)
        new_cols[f'Property{i}_min'] = df_original[props].min(axis=1)
        new_cols[f'Property{i}_std'] = df_original[props].std(axis=1)

    # Calculate Fraction stats
    frac_cols = [f"Component{i}_fraction" for i in range(1, 6)]
    new_cols["frac_sum"] = df_original[frac_cols].sum(axis=1)
    new_cols["frac_max"] = df_original[frac_cols].max(axis=1)
    new_cols["frac_min"] = df_original[frac_cols].min(axis=1)
    new_cols["frac_std"] = df_original[frac_cols].std(axis=1)

    # Combine all new columns with the original DataFrame at once
    return pd.concat([df_original, pd.DataFrame(new_cols)], axis=1)

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
        title="Fraction Sum Validation",
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
        title="Data Health Matrix",
        height=400,
        margin=dict(t=40, l=0, r=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, title="Features (Columns)"),
        yaxis=dict(showgrid=False, showticklabels=False, title="Rows"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240, 242, 246, 0.8)', # A slight off-white for the plot area
    )
    return fig

# --- Inverse_design Function ---
def inverse_design(target_properties, component_properties_row, assets, num_components=5):
    """
    Finds optimal fractions using a speed-tuned multi-start SLSQP approach.
    """
    prop_cols = [c for c in component_properties_row.columns if '_Property' in c]
    fixed_properties_df = component_properties_row[prop_cols]

    def objective_function(fractions):
        """The function to minimize: normalized squared error."""
        frac_df = pd.DataFrame([fractions], columns=[f'Component{i+1}_fraction' for i in range(num_components)])
        input_df = pd.concat([frac_df, fixed_properties_df.reset_index(drop=True)], axis=1)
        predicted_props_array = predict_properties(input_df, assets)
        
        error = 0.0
        for prop_name, target_value in target_properties.items():
            prop_index = int(prop_name.split('BlendProperty')[1]) - 1
            predicted_value = predicted_props_array[0, prop_index]
            blend_info = assets['all_models'][prop_name]['blend_info']
            prop_range = blend_info['max'] - blend_info['min']
            
            if prop_range > 1e-6:
                normalized_error = (predicted_value - target_value) / prop_range
            else:
                normalized_error = (predicted_value - target_value)
            error += normalized_error ** 2
        return error

    # --- Optimizer Configuration (Tuned for Speed) ---
    bounds = [(0, 1)] * num_components
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    optimizer_options = {'maxiter': 2500, 'ftol': 1e-4, 'disp': False} # CHANGED: Reduced maxiter

    # --- Multi-Start Optimization (Faster Version) ---
    n_starts = 10  
    best_result = None

    # Create a list of starting points
    initial_guesses = [np.ones(num_components) / num_components]
    for _ in range(n_starts - 1):
        random_guess = np.random.rand(num_components)
        initial_guesses.append(random_guess / random_guess.sum())

    # Run the optimizer from each starting point
    for guess in initial_guesses:
        result = minimize(
            objective_function,
            guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=optimizer_options
        )
        if result.success and (best_result is None or result.fun < best_result.fun):
            best_result = result
            
    # --- Process and Return the Best Result Found ---
    if best_result and best_result.success:
        final_fractions = best_result.x
        final_frac_df = pd.DataFrame([final_fractions], columns=[f'Component{i+1}_fraction' for i in range(num_components)])
        final_input_df = pd.concat([final_frac_df, fixed_properties_df.reset_index(drop=True)], axis=1)
        final_predictions = predict_properties(final_input_df, assets)
        return final_fractions, final_predictions[0], "Multi-start optimization successful."
    else:
        return None, None, "Optimization failed to converge from any starting point."

def render_metric_card(label, value, key):
    """Renders a styled card to display a label and a value."""
    st.markdown(f"""
    <div class="metric-card" id="metric-{key}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value:.4f}</div>
    </div>
    
    <style>
    .metric-card {{
        background-color: rgba(240, 242, 246, 0.7); /* Light grey with transparency */
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 5px solid #0072c6; /* A blue accent bar */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: #4F4F4F; /* Dark grey for the label */
        font-weight: 500;
    }}
    .metric-value {{
        font-size: 1.3rem;
        color: #013A63; /* Dark blue for the value */
        font-weight: 700;
    }}
    </style>
    """, unsafe_allow_html=True)

def plot_inverse_design_results(targets, predictions):
    """
    Generates a radar chart to visually compare target properties with the predicted properties.
    """
    prop_names = list(targets.keys())
    target_values = list(targets.values())
    pred_values = [predictions[int(p.split('BlendProperty')[1])-1] for p in prop_names]

    fig = go.Figure()

    # Trace for Achieved Properties
    fig.add_trace(go.Scatterpolar(
        r=pred_values, theta=prop_names, fill='toself', name='Achieved Properties',
        line=dict(color='#28a745'), fillcolor='rgba(40, 167, 69, 0.4)'
    ))

    # Trace for Target Properties
    fig.add_trace(go.Scatterpolar(
        r=target_values, theta=prop_names, fill='toself', name='Target Properties',
        line=dict(color='#0072c6'), fillcolor='rgba(0, 114, 198, 0.4)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor='#DDDDDD'),
            angularaxis=dict(tickfont=dict(size=11)), # Slightly larger font
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#222222"),
        height=350,  # FIX: Reduced height to better match the table
        legend=dict(
            orientation="h", # FIX: Horizontal legend is more compact
            yanchor="bottom",
            y=1.02, # FIX: Position it just above the plot area
            xanchor="center",
            x=0.5
        ),
        # FIX: Tighter margins, especially at the top
        margin=dict(l=40, r=40, t=40, b=40) 
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
    detail_color = "#32779C"     # Light Blue

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
    render_flow_block(
    "Feature Selection",
    "Reduces dimensionality with SHAP",
    "A separate LightGBM model is trained for each target solely to compute SHAP values. These models are different from the base LightGBM used later in stacking. The top-ranked features per target are retained to reduce noise and improve generalization.",
    "#DC2626",
    "🔍"
    )
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
    render_flow_block("Calibration","Isotonic Regression","Adjusts ensemble predictions using non-parametric Isotonic Regression, reducing systematic bias and aligning outputs closer to observed data for improved reliability.","#38BDF8","📈")
    st.markdown(f"<div style='text-align: center; margin-top: -12px; margin-bottom: -12px;'><img src='data:image/gif;base64,{gif_base64}' width='60' style='filter: invert(1);' /></div>", unsafe_allow_html=True)
    render_flow_block("Final Output","10 Optimized Predictions","Combines calibrated predictions with baseline weighted averages to produce accurate, reliable, and actionable estimates for 10 key blend properties.","#14B8A6","🎯")

def worker(args):
    prop, row_data, assets, component_to_vary = args
    return prop, run_sensitivity_analysis(row_data, assets, prop, component_to_vary)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def display_team_section():
    """
    Displays the new 'Meet the Team' section with a 2x2 grid and text block.
    """

    # --- Team Member Data (with image paths + zoom settings) ---
    team_members = [
        {"name": "Abhinav Tyagi", "role": "ML Engineer", "img": "images/Abhinav.jpg", "zoom": 1.8, "shift": 15},
        {"name": "Siddharth Bansal", "role": "Data Scientist", "img": "images/Siddharth.jpg", "zoom": 1.4, "shift": 8},
        {"name": "Shivang Sharma", "role": "Data Scientist", "img": "images/Shivang.jpeg", "zoom": 1.1, "shift": 0},
        {"name": "Utkarsh Singh", "role": "DevOps Engineer", "img": "images/Utkarsh.jpg", "zoom": 1.1, "shift": 0}
    ]

    # --- Helper: Encode local image to base64 ---
    def get_base64_image(image_path):
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None

    # --- Base CSS ---
    css_base = """
    <style>
        .team-container {
            display: flex;
            align-items: center;
            gap: 2rem;
        }
        .team-grid {
            flex: 1;
        }
        .team-text {
            flex: 1;
            padding-left: 2rem;
        }
        .team-card {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            height: 100%; 
            margin-bottom: 1.5rem;
        }
        .team-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.15);
        }
        .profile-img-container {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            margin: 0 auto 1rem auto;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(45deg, #f7b0c8, #b9e6ff);
            padding: 5px;
        }
        .profile-img {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background-color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .profile-img img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 50%;
        }
        .team-name {
            font-weight: 700;
            font-size: 1.2rem;
            color: #013A63;
            margin-bottom: 0.25rem;
        }
        .team-role {
            color: #005A9C;
            font-size: 0.9rem;
        }
        .team-text .small-header {
            font-weight: 600;
            color: #0072c6;
            margin-bottom: -0.5rem;
        }
        .team-text .big-header {
            font-size: 2.8rem;
            font-weight: 800;
            color: #004E92;
            margin-bottom: 1rem;
            line-height: 1.2;
        }
        .team-text p:not(.small-header) {
            color: #333;
            font-size: 1.1rem;
            line-height: 1.6;
        }
    </style>
    """

    # --- Dynamic CSS for each member (based on zoom/shift) ---
    css_dynamic = "<style>\n"
    for member in team_members:
        class_name = member["name"].split()[0].lower()  # e.g., abhinav, siddharth
        zoom = member.get("zoom", 1.0)
        shift = member.get("shift", 0)
        css_dynamic += f"""
        .profile-img img.{class_name} {{
            transform: scale({zoom}) translateY({shift}%);
            object-position: center;
        }}
        """
    css_dynamic += "</style>"

    # Inject both CSS blocks
    st.markdown(css_base + css_dynamic, unsafe_allow_html=True)

    # --- HTML Layout ---
    grid_col, text_col = st.columns([1.1, 1])

    with text_col:
        st.markdown("""
        <div class="team-text">
            <p class="small-header">Our Team</p>
            <h2 class="big-header">Locus</h2>
            <p>We are a passionate group of students—Abhinav, Siddharth, Shivang, and Utkarsh—united by our enthusiasm for machine learning and data science.</p>
            <p>Our diverse skills and collaborative spirit drive us to tackle complex challenges and build innovative, AI-powered solutions.</p>
            <p>And we all love coffee :D . </p>
        </div>
        """, unsafe_allow_html=True)

    with grid_col:
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        cols = [row1_col1, row1_col2, row2_col1, row2_col2]

        for i, member in enumerate(team_members):
            with cols[i]:
                img_b64 = get_base64_image(member["img"])
                img_src = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "https://via.placeholder.com/120"
                
                # assign class name based on first name
                class_name = member["name"].split()[0].lower()
                
                st.markdown(f"""
                <div class="team-card">
                    <div class="profile-img-container">
                        <div class="profile-img">
                            <img class="{class_name}" src="{img_src}" alt="{member['name']}">
                        </div>
                    </div>
                    <div class="team-name">{member['name']}</div>
                    <div class="team-role">{member['role']}</div>
                </div>
                """, unsafe_allow_html=True)

def display_footer():
    """
    Compact, full-bleed footer that matches the app theme,
    avoids horizontal scroll, and sits at the bottom (not fixed).
    """
    footer_css = """
    <style>
      /* Let the page push the footer to the bottom on short pages */
      .block-container {
          display: flex;
          flex-direction: column;
          min-height: 100vh;
          padding-bottom: 0 !important; /* remove extra bottom padding */
      }
      .flex-spacer { 
          flex: 1 0 auto; 
          min-height: 8rem; /* Add this line for a minimum space */
      }

      /* Kill the tiny horizontal scrollbar that full-bleed sections can cause */
      html, body, .stApp { overflow-x: hidden; }

      /* Full-bleed footer background (breaks out of Streamlit's centered layout) */
      .footer-bleed {
          position: relative;
          margin-left: calc(50% - 50vw);
          margin-right: calc(50% - 50vw);
          width: 99.5vw;
          border-top: 1px solid rgba(17,24,39,0.08);
          /* Match the app’s pastel theme with a soft white veil for contrast */
          background:
              linear-gradient(90deg, rgba(255,255,255,0.65), rgba(255,255,255,0.65)),
              linear-gradient(90deg, #f7b0c8 0%, #b9e6ff 100%);
          box-shadow: 0 -1px 0 rgba(0,0,0,0.04) inset;
      }

      /* Keep text area compact and centered */
      .footer-inner {
          max-width: 1200px;
          margin: 0 auto;
          padding: 0.75rem 1rem;      
          text-align: center;
          font-size: 1rem;         
          color: #0f172a;             
          line-height: 1.2;
      }
    </style>
    """
    footer_html = """
    <div class="flex-spacer"></div>
    <footer class="footer-bleed">
      <div class="footer-inner">
        © 2025 Locus · There's a 99% chance this was built after midnight ;)
      </div>
    </footer>
    """
    st.markdown(footer_css, unsafe_allow_html=True)
    st.markdown(footer_html, unsafe_allow_html=True)
def nav_to_top():
    """Injects JavaScript to scroll the window to the top."""
    js = """
    <script>
        // Set a timeout to run after the page has finished rendering
        setTimeout(function() {
            // Target the main window and scroll to the top
            window.parent.scrollTo(0, 0);
        }, 0);
    </script>
    """
    html(js, height=0)
# --- 3. Main Application ---
def main():
    st.set_page_config(page_title="Fuel Blend AI", layout="wide")
    st_javascript("window.scrollTo(0, 0);")

    # --- Gradient Background Theme ---
    st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem !important;
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

    # -- Gradient Button Theme
    st.markdown("""
    <style>
    /* Style all Streamlit buttons */
    .stButton > button {
    width: 100% !important;
    display: block !important;
    margin: 0 auto !important;
    padding: 0.65rem 1rem !important;
    border-radius: 12px !important;
    font-weight: 600 !important;

    /* Base: white pill with black text + gradient border */
    color: #111 !important;
    border: 3.5px solid transparent !important;  /* slightly thinner border */
    background:
        linear-gradient(#ffffff, #ffffff) padding-box,
        linear-gradient(90deg, #ffb3d9, #ff9999, #ffd36b) border-box !important; /* pink → light red → yellow */

    box-shadow: none !important;
    transition: background .2s ease, color .2s ease, transform .05s ease, box-shadow .2s ease;
    }

    /* Hover: invert theme — gradient fill, white border, white text */
    .stButton > button:hover {
    color: #ffffff !important;
    background:
        linear-gradient(90deg, #ffb3d9, #ff9999, #ffd36b) padding-box,
        linear-gradient(#ffffff, #ffffff) border-box !important;
    box-shadow: 0 6px 20px rgba(255, 179, 217, 0.25);
    }

    /* Press + accessibility */
    .stButton > button:active { transform: translateY(1px); }
    .stButton > button:focus-visible {
    outline: 3px solid rgba(255, 179, 217, 0.55) !important;
    outline-offset: 2px !important;
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
                border-radius: 50px!important;
                background-image: linear-gradient(45deg, #0057b7 30%, #00d4ff 70%)!important; 
                color: white !important;
                border: none!important;
                transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out!important;
                box-shadow: 0 4px 18px rgba(0, 212, 255, 0.35)!important;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.15)!important;
            }

            /* make sure the text inside also scales */
            div[data-testid="stButton"] > button > div > p,
            div[data-testid="stButton"] > button > span {
                font-size: 1.3em !important;
                font-weight: 800 !important;
            }

            div[data-testid="stButton"] > button:hover {
                transform: scale(1.05)!important;
                background-image: linear-gradient(45deg, #0057b7 30%, #00d4ff 70%)!important; 
                box-shadow: 0 8px 28px rgba(0, 212, 255, 0.45)!important;
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
                filter: brightness(0.95) drop-shadow(4px 6px 8px rgba(0, 0, 0, 0.4));
            }
            .logo-container:hover #interactive-logo-img {
                transform: scale(1.1);
                transition: transform 0.05s linear, filter 0.4s ease-out;
                filter: brightness(0.95) drop-shadow(6px 8px 12px rgba(0, 0, 0, 0.5));
            }
        </style>
        """, unsafe_allow_html=True)
        
        # --- Page Content ---
        with st.container():
            try:
                logo_path = "images/Shell Logo.png"
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
            st.markdown("""
            <div style="text-align:center; padding:2em 0; margin-top:2em;">
                <h2 style="color:#0072c6;">Are You Ready to Predict the Future of Fuel?</h2>
                <p style="color:#005A9C;">Step inside the AI-powered lab that helps design sustainable fuel blends at scale.</p>
            </div>
            """, unsafe_allow_html=True)
            col1, col2,col3 = st.columns([1,1,1])
            with col2:
                if st.button("Launch Prediction",use_container_width=True):
                    st.session_state.step = 1
                    streamlit_js_eval(js_expressions="window.scrollTo(0,0)")
                    st.rerun()
            st.markdown("""
                <style>
                hr {
                    border: none;       /* remove default border */
                    height: 2px;        /* thickness */
                    background-color: #0057b7;  /* blue color */
                }
                </style>
            """, unsafe_allow_html=True)
            st.markdown('---')
            st.markdown('<div class="section-header">How We Solve It</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                render_flow_block("Feature Engineering", "Creates derived features & weights.", "Automated creation of hundreds of insightful features that capture complex interactions between components.", "#3B82F6", "🧮",200)    
            with col2:
                render_flow_block("Model Stacking", "Combines strengths of multiple learners.", "We use a meta-learning approach, where a final model learns to optimally weigh the predictions from our base models.", "#6B7280", "🛠",200)
            with col3:
                render_flow_block("Calibrated Predictions", "Confidence-tuned ensemble outputs.", "Our models are calibrated to provide not just predictions, but a reliable measure of confidence, ensuring trustworthy results.", "#2ECC71", "📈",200)

            st.markdown('<div class="section-header">What Powers Our Predictions</div>', unsafe_allow_html=True)
            render_flow_diagram()


            # --- Meet theTeam Section ---
            st.markdown('<div class="section-header">Meet Our Team</div>', unsafe_allow_html=True)
            display_team_section() # ✨ This call renders the new design

        display_footer()
        return
    
        # --- CUSTOM STYLED HEADER (FIXED) ---
    try:
        logo_path = "images/Shell Logo.png"
        logo_base64 = image_to_base64(logo_path)
        logo_img_html = f'<div class="logo-bg"><img src="data:image/png;base64,{logo_base64}"></div>'
    except FileNotFoundError:
        logo_img_html = "" # Logo will be omitted if not found

    st.markdown(f"""
    <style>
        /* --- FIX: Hides the default Streamlit header, which blocks clicks on our custom header --- */
        [data-testid="stHeader"] {{
            display: none !important;
        }}

        .custom-header {{
            position: fixed; top: 0; left: 0; right: 0; width: 100%; z-index: 9999;
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-radius: 0 0 20px 20px;
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.6rem 1.5rem;
            height: 68px;
        }}

        /* Make link fill the left area and clickable across full height */
        .header-link {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
            height: 100%;
            text-decoration: none !important;
            color: inherit !important;
            padding: 0; 
            margin: 0;
        }}

        /* Logo background for visibility */
        .logo-bg {{
            background: rgba(255,255,255,0.8);
            border-radius: 50%;
            padding: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        }}
        .logo-bg img {{
            width: 42px;
            height: auto;
            display: block;
        }}

        .header-title {{
            font-size: 1.3rem;
            font-weight: 700;
            margin: 0;
            line-height: 1; /* ensures better vertical alignment */
            background: linear-gradient(to right, #0072c6 20%, #28a745 80%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header-team {{
            font-size: 1.8rem;
            font-weight: 500;
            color: #005A9C;
            margin-right: 2.5rem; /* space from right */
        }}

        /* Push content below fixed header */
        .block-container {{ padding-top: 5.5rem !important; }}
    </style>

    <div class="custom-header">
        <a href="." target="_self" class="header-link">
            {logo_img_html}
            <h2 class="header-title">Shell.ai Hackathon</h2>
        </a>
        <div class="header-team">Locus</div>
    </div>
    """, unsafe_allow_html=True)


    display_step_progress(st.session_state.step, mode="batch")
    # STEP 1: Upload CSV
    if st.session_state.step == 1:
        st.header("Step 1: Upload Batch File")

        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])
            st.markdown(
            """
            Your CSV file must have:
            - An `ID` column.
            - 5 `ComponentX_fraction` columns (X in 1-5).
            - 50 `ComponentX_PropertyY` columns(X in 1-5 and Y in 1-10).
            - The component fractions for each row must sum to 1.0.
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


            st.subheader("File Health Analysis")
            st.session_state.batch_input_df = final_df

            is_valid, msg = validate_batch_input(final_df)
            if not is_valid:
                st.error(f" Validation Failed :(  {msg}")
            else:
                st.success("Validation Passed :)  Your data looks good! You can now proceed to prediction.")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_missing_matrix(final_df), use_container_width=True)
                st.markdown(
                    '<p style="color:black; text-align: center;"><b>COACH\'S TIP:</b> This matrix shows missing data (red spots). A fully light-green chart is healthy!</p>',
                    unsafe_allow_html=True
                )
            with col2:
                st.plotly_chart(plot_fraction_sums(final_df), use_container_width=True)
                st.markdown(
                    '<p style="color:black; text-align: center;"><b>VALIDATION:</b> This chart checks if fractions sum to 1.0. Red bars show rows needing fixes.</p>',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            if st.button("Predict ➡", use_container_width=True, disabled=not is_valid):
                st.session_state.step = 2
                streamlit_js_eval(js_expressions="window.scrollTo(0,0)")
                st.rerun()
        display_footer()
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

            # ✅ Run predictions only once
            if "final_prediction_df" not in st.session_state:
                with st.spinner("🔄 Running batch predictions..."):
                    try:
                        predictions = predict_properties(df, assets)
                        pred_df = pd.DataFrame(
                            predictions, 
                            columns=[f"BlendProperty{i}" for i in range(1, 11)]
                        )
                        final_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
                        st.session_state.final_prediction_df = final_df
                    except Exception as e:
                        st.error(f" Error during prediction: {e}")
                        st.stop()

            final_df = st.session_state.final_prediction_df

            st.subheader("Prediction Results")

            # --- AGGRID TABLE ---
            cols_to_display = ["ID"] + [f"BlendProperty{i}" for i in range(1, 11)]
            display_df = final_df[cols_to_display]

            gb = GridOptionsBuilder.from_dataframe(display_df)
            gb.configure_default_column(
                filter=True,
                sortable=True,
                resizable=True
            )
            gb.configure_column("ID", width=80, minWidth=50, maxWidth=100)
            #  Explicitly ensure sorting is on for all columns
            for col in display_df.columns:
                gb.configure_column(col, sortable=True,headerClass="bold-header")
            gb.configure_grid_options(fitColumnsOnGridLoad=True)  # auto-adjust widths
            # gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            grid_options = gb.build()
            grid_options['defaultColDef']['sortable'] = True  # Double ensure sorting works
            grid_options['multiSortKey'] = 'ctrl'  # Enable multi-column sorting
            grid_options['rowHoverHighlight'] = True
            grid_options["floatingFilter"] = True
            grid_options["enableRangeSelection"] = True
            grid_options["enableCellTextSelection"] = True
            st.markdown("""
            <style>
            .ag-theme-streamlit .ag-header-cell.bold-header .ag-header-cell-label {
                font-weight: 700 !important;
            }
            </style>
            """, unsafe_allow_html=True)
            grid_response = AgGrid(
                display_df,
                gridOptions=grid_options,
                enable_enterprise_modules=False,
                fit_columns_on_grid_load=True,
                theme="streamlit",
                update_mode="MODEL_CHANGED",
                allow_unsafe_jscode=True
            )

            # --- DOWNLOAD (MATCHES FILTERED VIEW) ---
            filtered_df = pd.DataFrame(grid_response["data"])
            csv_to_download = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Results CSV",
                data=csv_to_download,
                file_name="blend_predictions_full.csv",
                mime="text/csv",
                use_container_width=False,
                key="download_full_csv",
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
                if st.button("Go to Analysis ➡", use_container_width=True):
                    st.session_state.step = 3
                    st.rerun()
        display_footer()
        
    elif st.session_state.step == 3:
        def sync_slider_to_num(prop_name):
            """Callback to update the slider's state from the number input."""
            st.session_state[f'slider_{prop_name}'] = st.session_state[f'num_{prop_name}']

        def sync_num_to_slider(prop_name):
            """Callback to update the number input's state from the slider."""
            st.session_state[f'num_{prop_name}'] = st.session_state[f'slider_{prop_name}']

        st.header("Step 3: Blend Analysis & Explainability")
        if "final_prediction_df" not in st.session_state:
            st.warning("⚠ No prediction data available. Please go back to Step 2.")
            if st.button("⬅ Back to Prediction Results"):
                st.session_state.step = 2
                st.rerun()
            return

        df = st.session_state.final_prediction_df

        # --- HORIZONTAL FULL-WIDTH TAB-LIKE SELECTOR ---
        # MODIFIED: Added a third column and button for Inverse Design
        col1, col2, col3 = st.columns([1, 1, 1])
        if "section" not in st.session_state:
            st.session_state.section = "Overall Dataset Analysis"

        with col1:
            if st.button("Overall Blend Analysis", key="btn_overall", use_container_width=True):
                st.session_state.section = "Overall Blend Analysis"
        with col2:
            if st.button("Single Blend Deep Dive", key="btn_single", use_container_width=True):
                st.session_state.section = "Single Blend Deep Dive"
        with col3:
            if st.button(" Inverse Blend Design", key="btn_inverse", use_container_width=True):
                st.session_state.section = "Inverse Blend Design"

        section = st.session_state.section
        st.markdown("---")

        # --- SHOW SELECTED SECTION ---
        if section == "Overall Blend Analysis":
            st.markdown(f"<h2>Overall Blend Analysis</h2>", unsafe_allow_html=True)
            # ... (your existing code for this section remains unchanged)
            numeric_df = df.select_dtypes(include='number')
            fraction_cols = [col for col in numeric_df.columns if 'fraction' in col]
            if fraction_cols:
                melted_frac = numeric_df[fraction_cols].melt(var_name="Component", value_name="Fraction")
                fig_box = px.box(
                    melted_frac, x="Component", y="Fraction", points="all", color="Component",
                    title="Distribution of Component Fractions Across All Uploaded Blends",
                    template="plotly_white"
                )
                fig_box.update_layout(
                    height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False, font=dict(color="#222222")
                )
                st.plotly_chart(fig_box, use_container_width=True)

            blend_props = [f"BlendProperty{i}" for i in range(1, 11) if f"BlendProperty{i}" in numeric_df.columns]
            if blend_props:
                melted_props = numeric_df[blend_props].melt(var_name="Property", value_name="Value")
                fig_props = px.box(
                    melted_props, x="Property", y="Value", points="all", color="Property",
                    title="Distribution of 10 BlendProperties", template="plotly_white"
                )
                fig_props.update_layout(
                    height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False, font=dict(color="#222222")
                )
                original_labels = melted_props['Property'].unique()
                new_labels = [label.replace('BlendProperty', 'Prop') for label in original_labels]
                fig_props.update_xaxes(tickvals=original_labels, ticktext=new_labels)
                st.plotly_chart(fig_props, use_container_width=True)

            st.markdown("<h3>Global Feature Importance (SHAP)</h3>", unsafe_allow_html=True)
            st.info(
                "This plot shows the most important features for a selected property across the entire uploaded dataset. "
                "Each point is a single prediction. Red points indicate that a high feature value pushed the prediction higher, "
                "while blue points indicate a low feature value pushed the prediction higher."
            )
            property_to_explain_global = st.selectbox(
                "Select a Blend Property to see its most influential features:",
                [f"BlendProperty{i}" for i in range(1, 11)],
                key="global_shap_property_selector"
            )
            if property_to_explain_global:
                with st.spinner(f"Generating global SHAP summary for {property_to_explain_global}..."):
                    input_data_for_shap = df.drop(columns=[f"BlendProperty{i}" for i in range(1, 11)])
                    generate_global_shap_summary(input_data_for_shap, property_to_explain_global, assets)


        elif section == "Single Blend Deep Dive":
            st.markdown(f"<h2>Single Blend Deep Dive</h2>", unsafe_allow_html=True)
            # ... (your existing code for this section remains unchanged)
            st.markdown("Select a single blend from your data to inspect its composition, understand its prediction, and run 'what-if' scenarios.")
            selected_id = st.selectbox("Select a Blend ID to analyze:", df["ID"].unique())
            row_data = df[df["ID"] == selected_id]
            st.subheader("Selected Row Composition")
            st.dataframe(row_data, use_container_width=True, height=80)
            st.subheader("Blend Composition Radars")
            col1, col2 = st.columns(2)
            with col1:
                components = [f"Component {i}" for i in range(1, 6)]
                frac_cols = [f"Component{i}_fraction" for i in range(1, 6)]
                fractions = [row_data.iloc[0][comp] for comp in frac_cols]
                fig_radar_frac = go.Figure()
                fig_radar_frac.add_trace(go.Scatterpolar(r=fractions + fractions[:1], theta=components + [components[0]], mode='lines', fill='toself', name='Fractions', line=dict(color="#0072c6"), fillcolor='rgba(0, 114, 198, 0.4)'))
                fig_radar_frac.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor='#DDDDDD'), angularaxis=dict(tickfont=dict(size=12), rotation=90), bgcolor='rgba(255, 255, 255, 0.5)'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#000000"), showlegend=False, height=400, title=dict(text="Component Fractions"))
                st.plotly_chart(fig_radar_frac, use_container_width=True)
            with col2:
                prop_labels = [f"Prop {i}" for i in range(1, 11)]
                prop_cols = [f"Component1_Property{i}" for i in range(1, 11)]
                prop_values = row_data.iloc[0][prop_cols].values.tolist()
                fig_radar_props = go.Figure()
                fig_radar_props.add_trace(go.Scatterpolar(r=prop_values + prop_values[:1], theta=prop_labels + [prop_labels[0]], mode='lines', fill='toself', name='Properties', line=dict(color="#28a745"), fillcolor='rgba(40, 167, 69, 0.4)'))
                fig_radar_props.update_layout(polar=dict(radialaxis=dict(visible=True, gridcolor='#DDDDDD'), angularaxis=dict(tickfont=dict(size=12)), bgcolor='rgba(255, 255, 255, 0.5)'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#000000"), showlegend=False, height=400, title=dict(text="Properties for Component 1"))
                st.plotly_chart(fig_radar_props, use_container_width=True)

            st.markdown("<h3>Prediction Explanation (Why?)</h3>", unsafe_allow_html=True)
            st.info(
                """
                What is Prediction Explanation? \n
                This tool helps you understand **why** the model made a specific prediction for a single blend composition. Instead of just giving you a number, it uses a technique called **SHAP** (SHapley Additive exPlanations) to break down the prediction and show you which features had the biggest influence.
            
                The plots you see, such as the **Force Plot**, **Waterfall Plot**, and **Decision Plot**, visualize this breakdown. They show how each individual feature's value contributed to pushing the prediction higher or lower than the average. This is crucial for building trust in the model and gaining actionable insights into what drives a blend's properties.
                """
            )
            col1, col2 = st.columns(2)
            with col1:
                property_to_explain = st.selectbox("Select a Blend Property to explain:", [f"BlendProperty{i}" for i in range(1, 11)], key="shap_property_selector")
            with col2:
                plot_type = st.selectbox("Select a plot type:", ["Force Plot", "Waterfall", "Decision Plot"], key="shap_plot_selector")
            if property_to_explain:
                with st.spinner(f"Generating {plot_type} for {property_to_explain}..."):
                    if plot_type == "Waterfall": generate_shap_waterfall_plot(row_data, property_to_explain, assets)
                    elif plot_type == "Decision Plot": generate_shap_decision_plot(row_data, property_to_explain, assets)
                    else: generate_shap_force_plot(row_data, property_to_explain, assets)

            st.markdown("<h3>Sensitivity Analysis (What If?)</h3>", unsafe_allow_html=True)
            st.info(
                "What is Sensitivity Analysis? \n\n"
                "This tool works backward. Instead of predicting properties from a known blend, you define the desired target properties for your outcome. The optimizer will then find the ideal component fractions required to achieve those targets, using one of your uploaded blends as a component baseline. \n"
                "Note: Due to limited computational power for backend, inverse prediction might take some time (30-50 sec)."
            )
            component_to_vary = st.selectbox("Select Component to Vary", [1, 2, 3, 4, 5], format_func=lambda x: f"Component {x}")
            if st.button("Run Sensitivity Analysis", use_container_width=True):
                # ... (rest of your sensitivity analysis code)
                blend_props = [f"BlendProperty{i}" for i in range(1, 11)]
                tasks = [(prop, row_data.copy(), assets, component_to_vary) for prop in blend_props]
                progress_bar = st.progress(0, text=" Launching parallel prediction threads...")
                results = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(worker, args): i for i, args in enumerate(tasks)}
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        prop, analysis_df = future.result()
                        results.append((prop, analysis_df))
                        progress_bar.progress((i + 1) / len(tasks), text=f" Completed {prop}...")
                progress_bar.empty()
                fig_sensitivity = go.Figure()
                for prop, analysis_df in sorted(results, key=lambda x: int(x[0].replace("BlendProperty", ""))):
                    fig_sensitivity.add_trace(go.Scatter(x=analysis_df['varied_fraction'], y=analysis_df['predicted_value'].astype(float), mode='lines+markers', name=prop))
                fig_sensitivity.update_layout(title=f"Sensitivity Analysis: Varying Component {component_to_vary}", xaxis_title=f"Fraction of Component {component_to_vary}", yaxis_title="Predicted Value", template="plotly_white", height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_sensitivity, use_container_width=True)

        # --- NEW INVERSE DESIGN SECTION ---
        elif section == "Inverse Blend Design":
            st.info(
                "**What is Inverse Design?** \n\n"
                "This tool works backward. Instead of predicting properties from a known blend, "
                "you define the desired target properties for your outcome. The optimizer will then "
                "find the ideal component fractions required to achieve those targets, using one of your "
                "uploaded blends as a component baseline. \n\n"
                "Note : Due to limited computational power for backend, inverse prediction might take some time (5-10 min)."
            )

            # --- Step 1: Select a Base Blend ---
            st.markdown("<h4>Step 1: Select Baseline Blend Properties</h4>", unsafe_allow_html=True)
            st.markdown("The optimizer needs a fixed set of properties for the 5 base components. Choose a blend from your uploaded data to serve as this baseline.")

            base_id = st.selectbox(
                "Select a Blend ID to use its component properties as the base:",
                df["ID"].unique(),
                key="inverse_design_base_id"
            )
            component_properties_row = df[df["ID"] == base_id].copy()

            # This is the correct and only location for this dataframe
            st.write("Full data for selected baseline:")
            st.dataframe(component_properties_row, use_container_width=True)

            st.markdown("---")

            # --- Permanent Box Displaying Baseline Fractions & Properties ---
            st.subheader(f"Baseline Properties for ID {base_id}")

            frac_col, chart_col = st.columns([1, 2])

            with frac_col:
                # ... (rest of the code for displaying metric cards remains the same)
                st.markdown("<h6>Baseline Fractions</h6>", unsafe_allow_html=True)
                for i in range(1, 6):
                    render_metric_card(
                        label=f"Component {i}",
                        value=component_properties_row[f'Component{i}_fraction'].iloc[0],
                        key=f"base_frac_card_{i}"
                    )

            with chart_col:
                # ... (rest of the code for the bar chart remains the same)
                st.markdown("<h6>Baseline Blend Properties</h6>", unsafe_allow_html=True)
                
                blend_props_series = component_properties_row[[f'BlendProperty{i}' for i in range(1, 11)]].iloc[0]
                plot_df = pd.DataFrame({
                    'Property': blend_props_series.index,
                    'Value': blend_props_series.values
                })

                fig = go.Figure()

                # --- New Code with Bar Borders ---
                fig.add_trace(go.Bar(
                    x=plot_df['Property'],
                    y=plot_df['Value'],
                    width= 0.4,
                    marker=dict(
                        color=['#1f77b4' if v >= 0 else '#636EFA' for v in plot_df['Value']], # Bar fill color
                        line=dict(
                            color='rgba(0, 0, 0, 0.8)', # Border color (dark and semi-transparent)
                            width=1                     # Border width
                        )
                    )
                ))
                
                # CHANGED: Set plot_bgcolor to transparent
                fig.update_layout(
                    yaxis_title="Value",
                    template="plotly_white",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', 
                    yaxis=dict(showgrid=True, gridcolor='rgba(220, 220, 220, 0.5)'),
                    xaxis=dict(showticklabels=True, tickangle=-45),
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- Step 2: Define Target Blend Properties ---
            st.markdown("---")
            st.markdown("<h4>Step 2: Define Target Blend Properties</h4>", unsafe_allow_html=True)
            st.markdown("Activate the properties you want to target, then adjust their values. The optimizer will only focus on the active ones.")

            targets = {}
            c1, c2 = st.columns(2)
            all_props = [f"BlendProperty{i}" for i in range(1, 11)]
            
            # --- New Code with 2-Way Sync ---
            for i, prop_name in enumerate(all_props):
                col = c1 if i < 5 else c2
                with col:
                    if st.checkbox(f"Target {prop_name}", value=False, key=f"check_{prop_name}"):
                        default_value = float(component_properties_row[prop_name].iloc[0])
                        default_value = np.clip(default_value, -3.0, 3.0)

                        # Initialize session state for both widgets if they don't exist
                        # This prevents them from resetting on every script run
                        if f'slider_{prop_name}' not in st.session_state:
                            st.session_state[f'slider_{prop_name}'] = default_value
                        if f'num_{prop_name}' not in st.session_state:
                            st.session_state[f'num_{prop_name}'] = default_value

                        slider_col, num_input_col = st.columns([0.7, 0.3])
                        
                        with slider_col:
                            st.slider(
                                f"Slider for {prop_name}",
                                min_value=-3.0,
                                max_value=3.0,
                                step=0.01,
                                key=f'slider_{prop_name}',
                                on_change=sync_num_to_slider, # Updates number input on change
                                args=(prop_name,),
                                label_visibility="collapsed"
                            )
                        
                        with num_input_col:
                            st.number_input(
                                "Value",
                                min_value=-3.0,
                                max_value=3.0,
                                step=0.01,
                                key=f'num_{prop_name}',
                                on_change=sync_slider_to_num, # Updates slider on change
                                args=(prop_name,),
                                label_visibility="collapsed"
                            )
                            
                        # The synchronized value can now be read from either state key
                        targets[prop_name] = st.session_state[f'num_{prop_name}']
            
            st.markdown("---")

            # --- Step 3: Run optimization on button press ---
            if st.button("Suggest Blend Composition", use_container_width=True, disabled=not targets):
                with st.spinner(" Running inverse design optimization... this might take some time."):
                    try:
                        fractions, preds, msg = inverse_design(targets, component_properties_row, assets)
                        if fractions is not None:
                            st.session_state.inverse_design_results = {
                                "fractions": fractions, "predictions": preds,
                                "targets": targets, "message": msg
                            }
                        else:
                            st.error(f"Optimization failed: {msg}")
                            if 'inverse_design_results' in st.session_state: del st.session_state['inverse_design_results']
                    except Exception as e:
                        st.error(f"An unexpected error occurred during optimization: {e}")
                        if 'inverse_design_results' in st.session_state: del st.session_state['inverse_design_results']
            
            # --- Step 4: Display results if they exist in session state ---
            if 'inverse_design_results' in st.session_state and st.session_state.inverse_design_results:
                results = st.session_state.inverse_design_results
                st.success(f"Optimization complete! Status: {results['message']}")
                st.markdown("<br>", unsafe_allow_html=True) # Adds a little space

                # --- ROW 1: Table and Radar Chart ---
                st.markdown("<h5>Comparison of Target vs. Achieved Properties</h5>", unsafe_allow_html=True)
                res1_col1, res1_col2 = st.columns([1.2, 1])

                with res1_col1:
                    # Build the comparison dataframe
                    comparison_data = []
                    for prop, target_val in results['targets'].items():
                        prop_idx = int(prop.split('BlendProperty')[1]) - 1
                        pred_val = results['predictions'][prop_idx]
                        absolute_error = abs(pred_val - target_val)
                        comparison_data.append({
                            "Property": prop,
                            "Target Value": target_val,
                            "Achieved Value": pred_val,
                            "Absolute Error": absolute_error
                        })
                    df_comparison = pd.DataFrame(comparison_data)

                    # Display the formatted dataframe
                    st.dataframe(
                        df_comparison.style.format({
                            "Target Value": "{:.4f}",
                            "Achieved Value": "{:.4f}",
                            "Absolute Error": "{:.4f}"
                        }),
                        use_container_width=True,
                        height=350
                    )

                with res1_col2:

                    # Display the radar chart
                    st.plotly_chart(
                        plot_inverse_design_results(results['targets'], results['predictions']),
                        use_container_width=True
                    )

                st.markdown("---")

                # --- ROW 2: Fractions Text and Pie Chart ---
                st.markdown("<h5>Suggested Optimal Blend Composition</h5>", unsafe_allow_html=True)
                res2_col1, res2_col2 = st.columns([1.2, 1])

                with res2_col1:
                    # Display the optimal fractions using styled metric cards
                    for i, fraction in enumerate(results['fractions']):
                        render_metric_card(
                            label=f"Component {i+1} Fraction",
                            value=fraction,
                            key=f"final_frac_card_{i}"
                        )

                with res2_col2:
                    # Create and display the pie chart
                    frac_df = pd.DataFrame({
                        "Component": [f"Component {i+1}" for i in range(5)],
                        "Fraction": results['fractions']
                    })
                    fig_pie = px.pie(frac_df, values='Fraction', names='Component', hole=0.3)
                    fig_pie.update_traces(textinfo='percent+label', textfont_size=14)
                    fig_pie.update_layout(
                        title_text='Optimal Component Fractions',
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=40, b=0, l=0, r=0)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True) 

        # --- Bottom Navigation ---
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅ Back to Prediction Results", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button(" Upload Another File", use_container_width=True):
                for key in ["batch_input_df", "final_prediction_df", "inverse_design_results"]:
                    st.session_state.pop(key, None)
                st.session_state.step = 1
                st.rerun()
        display_footer()

if __name__ == "__main__":
    main()
