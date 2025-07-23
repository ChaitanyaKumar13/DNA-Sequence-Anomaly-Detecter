import streamlit as st

# ✅ MUST BE FIRST Streamlit command
st.set_page_config(page_title="DNA Sequence Classifier", layout="centered")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.load_transformer_data import load_dataset
from transformer_dna_classifier import PositionalEncoding
import keras.saving

# ✅ Register custom PositionalEncoding layer
keras.saving.register_keras_serializable()(PositionalEncoding)

# ✅ Config
MODEL_PATH = "models/transformer_dna_uncertain.keras"
MAX_LEN = 180
CLASS_NAMES = ["EI", "IE", "Neither"]

# ✅ Load model once
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH, compile=False, custom_objects={'PositionalEncoding': PositionalEncoding})

model = load_trained_model()

# ✅ Monte Carlo Dropout + Temperature Scaling with unhashable fix
@st.cache_data
def mc_dropout_predict(_model, _X, n_iter=30, temp=0.8):
    f_model = tf.keras.Model(_model.input, _model.output)
    preds = np.stack([
        tf.nn.softmax(f_model(_X, training=True) / temp).numpy() for _ in range(n_iter)
    ])
    mean_preds = preds.mean(axis=0)
    std_preds = preds.std(axis=0).mean(axis=1)
    flags = (np.max(mean_preds, axis=1) < 0.55).astype(int)
    return mean_preds, std_preds, flags

# ✅ App UI
st.title("🧬 Transformer-based DNA Sequence Anomaly Classifier")
st.write("Upload a DNA CSV file to classify sequences and detect anomalies based on model confidence.")

# ✅ Upload file
uploaded_file = st.file_uploader("Upload a DNA CSV file", type=["csv"])

# ✅ Load & preprocess
@st.cache_data
def preprocess_data(file):
    return load_dataset(file, max_len=MAX_LEN)

# ✅ Process if uploaded
if uploaded_file:
    X, y = preprocess_data(uploaded_file)
    y = np.array(y).astype(int)
    X_tensor = tf.convert_to_tensor(X)

    # ✅ Predict
    y_probs, y_stddev, anomaly_flags = mc_dropout_predict(model, X_tensor)
    y_preds = np.argmax(y_probs, axis=1)

    confidences = np.max(y_probs, axis=1)
    labels = [CLASS_NAMES[i] for i in y_preds]
    status_ui = ["⚠️ Anomaly" if flag else "Normal" for flag in anomaly_flags]
    status_csv = ["Anomaly" if flag else "Normal" for flag in anomaly_flags]

    # ✅ Results DataFrame
    df_preds_ui = pd.DataFrame({
        "True Label": [CLASS_NAMES[i] for i in y],
        "Predicted Label": labels,
        "Confidence": np.round(confidences, 3),
        "Uncertainty": np.round(y_stddev, 3),
        "Status": status_ui
    })

    df_preds_csv = pd.DataFrame({
        "True Label": [CLASS_NAMES[i] for i in y],
        "Predicted Label": labels,
        "Confidence": np.round(confidences, 3),
        "Uncertainty": np.round(y_stddev, 3),
        "Status": status_csv
    })

    # ✅ Highlight anomalies in red
    def highlight_anomalies(row):
        return ['background-color: #ffcccc' if row["Status"] == "⚠️ Anomaly" else '' for _ in row]

    # ✅ Show prediction summary
    st.subheader("🔍 Prediction Summary with Anomaly Flags")
    st.dataframe(df_preds_ui.style.apply(highlight_anomalies, axis=1), use_container_width=True)

    # ✅ Download button (emoji-free status)
    st.subheader("⬇️ Download Predictions")
    csv = df_preds_csv.to_csv(index=False, encoding='utf-8-sig')
    st.download_button("Download Prediction CSV", csv, file_name="dna_predictions.csv")

else:
    st.warning("📂 Please upload a DNA CSV file to begin classification.")
