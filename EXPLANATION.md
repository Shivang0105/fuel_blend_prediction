Explanation of Our Approach: Shell.ai Hackathon Fuel Blend Modeling

Introduction

This document outlines our methodology for predicting ten physicochemical blend properties for sustainable aviation fuel (SAF) samples. Our approach integrates chemical intuition with modern machine learning techniques to provide accurate and physically meaningful property estimates for new blends.

---

Data Preparation

- The training, test, and sample submission files are loaded.
- The "ID" column is removed from feature data, and test IDs are set aside for the final submission.
- The target columns are BlendProperty1 to BlendProperty10.

---

Feature Engineering

Our feature engineering captures both physical blending principles and complex inter-component interactions:

1. Volume-Weighted Physical Baseline:  
   For each blend property, a baseline is computed as the weighted sum of each component's property and its blend fraction, following linear chemical blending laws.

2. Residual Features for Nonlinearity:  
   For each component and property, residuals are calculated by subtracting the blend weighted baseline from the component’s property. This captures nonlinear mixing effects.

3. Component Statistics:  
   For every component, the mean and standard deviation across all ten properties are computed. This characterizes the distribution of properties within each component.

4. Interaction Features:  
   For each property and component, the product of its fraction and property value is included, enabling the model to learn interaction effects based on blend composition.

5. Per-Property Statistics:  
   For each property, aggregate statistics (min, max, standard deviation) across components are calculated to capture blend diversity and extremes.

6. Fraction Statistics:  
   Summary statistics (sum, max, min, std) of the component fractions highlight any dominance or balance issues within the blend.

Features are generated for both training and test sets, aligned, missing values are imputed with zeros, and all features are scaled to [0, 1] using MinMaxScaler to support model training.

---

Neural Network Ensemble

An ensemble class comprises multiple (typically three) multi-layer perceptron (MLP) models, each trained with early stopping and adaptive learning rates.
The ensemble prediction is the average of the constituent neural networks, increasing the robustness and stability of this model type.

---

Model Training and Stacking

A comprehensive stacked model strategy is employed:

1. Target-Wise Processing:  
   Each property is modeled independently, with target normalization applied for training stability.

2. SHAP-Based Feature Selection:  
   LightGBM models are used to compute SHAP values, and the top 30 most important features are retained for each property. This focuses learning on the most relevant data and promotes interpretability.

3. Cross-Validation and Ensemble Training:  
   Multiple random seeds and four-fold cross-validation are used for reliable out-of-fold (OOF) predictions. Four base models are trained per fold: LightGBM, XGBoost, CatBoost, and the neural network ensemble.

4. Model Stacking with Ridge Regression:  
   OOF predictions from all base models are stacked, and a RidgeCV linear meta-model is fit to combine their outputs effectively.

5. Calibration and Post-Processing:  
   a. Ensemble predictions are de-normalized.
   b. Isotonic regression calibrates results to correct bias and ensure monotonicity.
   c. Predictions are blended with the physical baseline (using an optimized blending parameter, alpha) to enforce physical plausibility.
   d. Final results are clipped to observed value ranges to prevent extrapolation.

6. Ensemble Averaging:  
   Predictions are averaged across folds and random seeds to further enhance stability and robustness.

---

Submission

Final predictions are joined with their test IDs and saved in the required format for leaderboard evaluation.

---

Summary

Our pipeline leverages strong chemical baselines, effective and interpretable feature engineering, a diverse machine learning ensemble, model stacking, and calibration. These elements together ensure our predictions are accurate, robust, and consistent with real physical blending behavior, making them suitable for practical application in sustainable fuel formulation.

Locus

