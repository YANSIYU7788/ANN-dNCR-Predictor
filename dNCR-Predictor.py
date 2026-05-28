import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# =========================
# 1. Load model, scaler, feature columns, and thresholds
# =========================
model_path = "ann_calibrated_model.pkl"
scaler_path = "ann_scaler.pkl"
feature_cols_path = "ann_feature_columns.pkl"

model = joblib.load("ann_calibrated_model.pkl")
scaler = joblib.load(scaler_path)
feature_cols = joblib.load(feature_cols_path)

# 直接使用训练时计算得到的 ANN 最优阈值（0.23）
ann_threshold = 0.23
# =========================
# 2. Streamlit UI
# =========================
st.title("ANN-based dNCR Prediction")
st.write("Please enter patient characteristics:")

# Continuous features that need scaling
scale_cols = ['Age', 'MOCA_Score', 'Operation_Time', 'GFR']
input_data_original = {}

# Continuous features input
for col in scale_cols:
    input_data_original[col] = st.number_input(f"{col}:", value=0.0, step=0.1, format="%.1f")

# ADL (multicategorical) – map to numeric score
adl_options = {
    "Completely independent (100)": 100,
    "Mildly dependent (61-99)": 80,
    "Moderately dependent (41-60)": 50,
    "Severely dependent (21-40)": 30,
    "Completely dependent (≤20)": 10
}
adl_choice = st.selectbox("ADL (Activities of Daily Living):", list(adl_options.keys()))
input_data_original['ADL'] = adl_options[adl_choice]

# ASA grade (only II and III; I is reference)
asa_choice = st.selectbox("ASA grade:", ["II", "III"])
input_data_original['ASA'] = asa_choice
input_data_original['ASA_3'] = 1 if asa_choice == "III" else 0

# Depression
depression_choice = st.radio("Depression:", ("No", "Yes"))
input_data_original['Depression'] = depression_choice
input_data_original['Depression_1'] = 1 if depression_choice == "Yes" else 0

# =========================
# 3. Prediction button
# =========================
if st.button("Predict"):
    # Prepare input data for the model
    input_data = {}

    # Continuous features
    for col in scale_cols:
        input_data[col] = input_data_original[col]

    # ADL score
    input_data['ADL'] = input_data_original['ADL']

    # ASA_3 (dummy for ASA grade III)
    input_data['ASA_3'] = input_data_original['ASA_3']

    # Depression_1 (dummy for depression)
    input_data['Depression_1'] = input_data_original['Depression_1']

    # Ensure all feature columns are present (fill missing with 0)
    for col in feature_cols:
        if col not in input_data:
            input_data[col] = 0

    # Create DataFrame
    X_input = pd.DataFrame([input_data], columns=feature_cols)

    # Scale the continuous features
    X_input[scale_cols] = scaler.transform(X_input[scale_cols])

    # Prediction
    pred_prob = model.predict_proba(X_input)[:, 1]
    pred_label = (pred_prob >= ann_threshold).astype(int)

    st.write(f"Predicted probability: {pred_prob[0]:.4f}")
    st.write(f"Predicted result: {'Yes' if pred_label[0] == 1 else 'No'}")

    # =========================
    # Background data for SHAP (all zeros, then scaled)
    # =========================
    background_data = {}

    # Continuous features set to 0
    for col in scale_cols:
        background_data[col] = 0

    # Other features set to 0
    background_data['ADL'] = 0
    background_data['ASA_3'] = 0
    background_data['Depression_1'] = 0
    for col in feature_cols:
        if col not in background_data:
            background_data[col] = 0

    background_df = pd.DataFrame([background_data], columns=feature_cols)
    background_df[scale_cols] = scaler.transform(background_df[scale_cols])

    # =========================
    # SHAP explainer
    # =========================
    def predict_fn(x):
        x_df = pd.DataFrame(x, columns=feature_cols)
        return model.predict_proba(x_df)[:, 1]

    explainer = shap.KernelExplainer(predict_fn, background_df)
    shap_values = explainer.shap_values(X_input, nsamples=100)

    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]

    # =========================
    # Display SHAP values
    # =========================
    display_features = ['Age', 'MOCA_Score', 'Operation_Time', 'GFR', 'ADL', 'ASA', 'Depression']

    st.write("SHAP values for each feature:")
    for feat in display_features:
        if feat == 'ADL':
            idx = feature_cols.index('ADL')
            st.write(f"{feat}: {shap_values[idx]:.4f} (value = {input_data_original['ADL']})")
        elif feat == 'ASA':
            idx = feature_cols.index('ASA_3')
            st.write(f"{feat}: {shap_values[idx]:.4f} (value = {input_data_original['ASA']})")
        elif feat == 'Depression':
            idx = feature_cols.index('Depression_1')
            st.write(f"{feat}: {shap_values[idx]:.4f} (value = {input_data_original['Depression']})")
        elif feat in feature_cols:
            idx = feature_cols.index(feat)
            st.write(f"{feat}: {shap_values[idx]:.4f} (value = {input_data_original[feat]})")
        else:
            st.write(f"{feat}: not found in model")

    # =========================
    # SHAP force plot
    # =========================
    st.subheader("SHAP Force Plot")

    shap_vals_list = []
    feature_vals_list = []
    feature_names_list = []

    for feat in display_features:
        if feat == 'ADL':
            idx = feature_cols.index('ADL')
            shap_vals_list.append(shap_values[idx])
            feature_vals_list.append(input_data_original['ADL'])
            feature_names_list.append(feat)
        elif feat == 'ASA':
            idx = feature_cols.index('ASA_3')
            shap_vals_list.append(shap_values[idx])
            feature_vals_list.append(input_data_original['ASA'])
            feature_names_list.append(feat)
        elif feat == 'Depression':
            idx = feature_cols.index('Depression_1')
            shap_vals_list.append(shap_values[idx])
            feature_vals_list.append(input_data_original['Depression'])
            feature_names_list.append(feat)
        elif feat in feature_cols:
            idx = feature_cols.index(feat)
            shap_vals_list.append(shap_values[idx])
            feature_vals_list.append(input_data_original[feat])
            feature_names_list.append(feat)

    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=np.array(shap_vals_list),
        features=np.array(feature_vals_list),
        feature_names=feature_names_list,
        matplotlib=False
    )

    shap.save_html("shap_force_plot.html", force_plot)

    with open("shap_force_plot.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=400, scrolling=False)