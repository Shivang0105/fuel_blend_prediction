
# Shell.ai Hackathon 2025 – Fuel Blend Properties Prediction

Welcome to our solution for the **Shell.ai Hackathon for Sustainable and Affordable Energy 2025**. This repository contains an end-to-end system for predicting and optimizing properties of sustainable fuel blends using advanced machine learning and explainable AI.

---

## 🚩 Project Overview

The goal is to develop robust AI models that can:
- Predict final blend properties of complex fuels from their component fractions and properties.
- Enable inverse design: suggesting optimal blend compositions for user-specified property targets.
- Provide transparent, interpretable insights through advanced explainability tools.

---

## 📁 Directory Structure
```bash
.
├── .devcontainer/              # Dev container configuration
├── .streamlit/                 # Streamlit app configuration
├── .vscode/                    # IDE settings for Python compatibility
├── catboost_info/              # CatBoost model artifacts/info
├── datasets/                   # Dataset files (train/test/sample_submission)
├── images/                     # Project-related images and visualizations
├── models/                     # Trained model files and artifacts
├── output/                     # Prediction outputs and logs
├── feature_engineering.py      # Feature engineering logic
├── model_training.py           # ML model training pipeline
├── nn_model.py                 # Neural network ensemble implementation
├── predict_blend_properties.py # Batch prediction script
├── streamlit_app_v2.py         # Interactive Streamlit dashboard
├── requirements.txt            # Python dependencies
└── EXPLANATION.md              # Full methodology and technical documentation

```

---
## 🚀 Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/Shivang0105/fuel_blend_prediction.git
cd fuel_blend_prediction
```

```bash
2. **Install dependencies**
pip install -r requirements.txt
```

```bash
3. **(Optional) Set up development container**  
Use `.devcontainer` or `.vscode` folders for rapid environment configuration.
```
---

## 🛠️ Usage

- **Batch Prediction:**  
Run property predictions using `predict_blend_properties.py` with your blend data.

- **Interactive Dashboard:**  
Launch

```bash
streamlit run streamlit_app_v2.py
```

Upload your blend input file, analyze global/local explainability, and try Inverse Blend Design for custom property targets.

---

## 📊 Key Features

- **Advanced Feature Engineering:** Captures complex non-linearities via domain-inspired features.
- **Ensemble Modeling:** Combines LightGBM, XGBoost, CatBoost, and neural network predictions.
- **Model Explainability:** Integrated SHAP explanations at both global and local levels.
- **Inverse Blend Design:** Optimizes input fractions to achieve user-defined property goals.
- **Rich UI:** Interactive data validation, predictions table, visual analysis, and export options.

---

## 🧑‍💻 Team

A multidisciplinary team passionate about ML, AI, and sustainable energy solutions.

---

## 📄 Documentation

For a deep dive into methodology, model architecture, and examples, see [`EXPLANATION.md`](EXPLANATION.md).

---

## 🤝 Contributing

We welcome improvements and feedback!  
Feel free to fork, open pull requests, or raise issues.

---

## 📝 License

This project is open-sourced under the MIT License.

---

*Built for the Shell.ai Hackathon for Sustainable and Affordable Energy 2025.*
