import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import os
import io
import pywt
import matplotlib.pyplot as plt

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CardioVision AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    h1 {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        font-size: 2.4rem !important;
    }
    h2 {
        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600 !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #6e8efb 0%, #a777e3 100%);
        color: white !important;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 10px !important;
        border: none;
        padding: 0.6rem 2rem;
        box-shadow: 0 4px 15px rgba(110, 142, 251, 0.35);
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(110, 142, 251, 0.5);
    }

    div.stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        font-weight: 600;
        border-radius: 10px !important;
        border: none;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.35);
        transition: all 0.3s ease;
    }
    div.stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.5);
    }

    img {
        border-radius: 10px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }
    img:hover { transform: scale(1.015); }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #4ECDC4 !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6e8efb, #a777e3) !important;
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #4ECDC4, transparent);
        margin: 1.5rem 0;
    }

    /* ─── Custom Navigation Style (Radio Button Override) ─── */
    
    /* Hide the default radio circles completely */
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-of-type {
        display: none !important;
    }
    
    /* Container for the nav items */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.5rem;
    }
    
    /* Menu item text styling */
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.5) !important;
        margin: 0;
        padding: 5px 10px;
        transition: all 0.2s ease-in-out;
    }
    
    /* Hover state */
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover p {
        color: rgba(255, 255, 255, 0.9) !important;
        transform: translateX(5px);
    }
    
    /* Active/Checked state */
    /* Streamlit changes background or font color on checked, we use important to override */
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
    [data-testid="stSidebar"] div[role="radiogroup"] div[aria-checked="true"] p,
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff !important;
        transform: translateX(8px);
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────
CLASSIFIER_PATH  = "resnet_classifier.keras"
GENERATOR_PATH   = "gan_generator.keras"
AUTOENCODER_PATH = "autoencoder.keras"
LATENT_DIM = 128

# Google Drive file IDs for auto-download
GDRIVE_FILES = {
    CLASSIFIER_PATH:  "1GJytjOy13jA-onsE3dLYGSv8ExcyQYRx",
    GENERATOR_PATH:   "1b2e9FOIE3fNkBQJ3Z6S3zhdlO_BiebDV",
    AUTOENCODER_PATH: "1xLpQl0bogvbZpmqUW-29KtTuqqi_S6CD",
}

def download_models():
    """Download model files from Google Drive if they don't exist locally."""
    import gdown
    for filename, file_id in GDRIVE_FILES.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            with st.spinner(f"Downloading {filename}... (first launch only)"):
                gdown.download(url, filename, quiet=False)

download_models()

CLASSES = {
    0: 'Normal (N)', 1: 'Supraventricular (S)',
    2: 'Ventricular (V)', 3: 'Fusion (F)', 4: 'Unknown (Q)'
}
CLASS_EMOJI = {
    0: '🟢', 1: '🔵', 2: '�', 3: '🟠', 4: '🟣'
}

# Sample heartbeat signal (Normal sinus rhythm)
SAMPLE_SIGNAL = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01,
    0.02, 0.04, 0.08, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00, 0.85,
    0.55, 0.20, -0.10, -0.15, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.05, 0.10, 0.15, 0.18, 0.20, 0.18, 0.15, 0.10, 0.05, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
], dtype=np.float32)

# ─── Model Loading ─────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path, compile=False)
    return None

# ─── Utilities ─────────────────────────────────────────────────
def preprocess(image, size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def signal_to_cwt(signal_1d):
    widths = np.arange(1, 128)
    coeffs, _ = pywt.cwt(signal_1d, widths, 'mexh')
    coeffs = np.abs(coeffs)
    coeffs = (coeffs - coeffs.min()) / (coeffs.max() - coeffs.min() + 1e-8)
    coeffs = (coeffs * 255).astype(np.uint8)
    img = cv2.applyColorMap(coeffs, cv2.COLORMAP_VIRIDIS)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(img)

def array_to_png_bytes(arr):
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding-bottom: 20px;'>
        <h1 style='font-size: 2.2rem; margin-bottom: 0;'>🫀</h1>
        <h2 style='font-size: 1.5rem; margin-top: 0; background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>CardioVision AI</h2>
        <p style='color: #888; font-size: 0.9rem; margin-top: -10px;'>Arrhythmia Analysis Suite</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<p style='font-size: 0.85rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>Navigation</p>", unsafe_allow_html=True)
    page = st.radio("", [
        "Classification",
        "Batch Prediction",
        "Synthetic Generation",
        "Anomaly Detection",
        "About Project"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    # ── Minimalist Footer Profile ──
    st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <p style='color: #888; font-size: 0.8rem; margin: 0;'>POWERED BY</p>
        <img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" width="100" style="margin-top: 5px; opacity: 0.8;">
        <br><br>
        <p style='color: #666; font-size: 0.75rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;'>v1.0.0 Release</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE 1: CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
if "Classification" in page:
    st.title("Arrhythmia Classification")
    st.markdown("Classify ECG signals into 5 arrhythmia categories using a fine-tuned ResNet50 model.")
    st.markdown("---")

    classifier = load_model(CLASSIFIER_PATH)

    if classifier is None:
        st.error("Classification model not found. Place `resnet_classifier.keras` in the app directory.")
    else:
        input_type = st.radio("Input Type", ["CWT Image", "Raw CSV Signal", "Try Sample Data"], horizontal=True)

        image = None

        if input_type == "CWT Image":
            uploaded = st.file_uploader("Upload a CWT spectrogram image", type=["png", "jpg", "jpeg"])
            if uploaded:
                image = Image.open(uploaded)

        elif input_type == "Raw CSV Signal":
            uploaded = st.file_uploader("Upload a heartbeat CSV file", type=["csv"])
            if uploaded:
                df = pd.read_csv(uploaded, header=None)
                st.caption(f"📊 {len(df)} heartbeats detected in file")
                row = st.number_input("Select heartbeat index", 0, len(df)-1, 0)
                signal = df.iloc[row, :187].values
                st.line_chart(signal, use_container_width=True)
                image = signal_to_cwt(signal)

        else:  # Sample Data
            st.info("Using a built-in sample ECG heartbeat signal for demonstration.")
            st.line_chart(SAMPLE_SIGNAL, use_container_width=True)
            image = signal_to_cwt(SAMPLE_SIGNAL)

        if image is not None:
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.image(image, caption="Input Image", use_container_width=True)

            with col2:
                with st.spinner("Analyzing..."):
                    processed = preprocess(image)
                    preds = classifier.predict(processed, verbose=0)
                    idx = int(np.argmax(preds))
                    conf = float(np.max(preds))

                st.metric("Predicted Class", f"{CLASS_EMOJI[idx]} {CLASSES[idx]}")
                st.metric("Confidence", f"{conf*100:.1f}%")

                st.markdown("**Class Probabilities**")
                for i, p in enumerate(preds[0]):
                    st.progress(float(p), text=f"{CLASS_EMOJI[i]} {CLASSES[i]}: {p*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
#  PAGE 2: BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.title("Batch Prediction")
    st.markdown("Upload a CSV file with multiple heartbeats and classify all of them at once.")
    st.markdown("---")

    classifier = load_model(CLASSIFIER_PATH)

    if classifier is None:
        st.error("Classification model not found. Place `resnet_classifier.keras` in the app directory.")
    else:
        uploaded = st.file_uploader("Upload a heartbeat CSV file", type=["csv"], key="batch")

        if uploaded:
            df = pd.read_csv(uploaded, header=None)
            total = len(df)
            st.caption(f"📊 {total} heartbeats detected")

            max_rows = st.slider("Number of heartbeats to classify", 1, min(total, 500), min(total, 100))

            if st.button(f"Classify {max_rows} Heartbeats"):
                progress_bar = st.progress(0, text="Starting batch classification...")
                results = []

                for i in range(max_rows):
                    signal = df.iloc[i, :187].values
                    cwt_img = signal_to_cwt(signal)
                    processed = preprocess(cwt_img)
                    preds = classifier.predict(processed, verbose=0)
                    pred_class = int(np.argmax(preds))
                    confidence = float(np.max(preds))
                    results.append({
                        "Index": i,
                        "Predicted Class": CLASSES[pred_class],
                        "Confidence (%)": round(confidence * 100, 1)
                    })
                    if (i + 1) % 5 == 0 or i == max_rows - 1:
                        progress_bar.progress((i + 1) / max_rows, text=f"Classified {i+1}/{max_rows}")

                progress_bar.empty()
                results_df = pd.DataFrame(results)

                # Summary
                st.markdown("### Results Summary")
                counts = results_df["Predicted Class"].value_counts()

                col1, col2 = st.columns([1, 1], gap="large")
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
                    counts.plot(kind='bar', ax=ax, color=colors[:len(counts)], edgecolor='none')
                    ax.set_title("Class Distribution", fontsize=14, fontweight='bold')
                    ax.set_ylabel("Count")
                    ax.set_xlabel("")
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    fig2, ax2 = plt.subplots(figsize=(5, 5))
                    ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                            colors=colors[:len(counts)], startangle=90,
                            textprops={'fontsize': 9})
                    ax2.set_title("Class Distribution", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig2)

                # Full results table
                st.markdown("### Detailed Results")
                st.dataframe(results_df, use_container_width=True, height=300)

                # Download results
                csv_bytes = results_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv_bytes,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )

# ═══════════════════════════════════════════════════════════════
#  PAGE 3: GENERATION
# ═══════════════════════════════════════════════════════════════
elif "Generation" in page:
    st.title("Synthetic ECG Generation")
    st.markdown("Generate synthetic CWT spectrograms from random latent vectors using a trained DCGAN.")
    st.markdown("---")

    generator  = load_model(GENERATOR_PATH)
    classifier = load_model(CLASSIFIER_PATH)

    if generator is None:
        st.error("Generator model not found. Place `gan_generator.keras` in the app directory.")
    else:
        num = st.slider("Number of images", 1, 5, 3)

        if st.button("Generate"):
            with st.spinner("Synthesizing..."):
                noise = np.random.normal(0, 1, (num, LATENT_DIM)).astype(np.float32)
                imgs = generator.predict(noise, verbose=0)
                imgs = np.clip((imgs + 1) / 2.0, 0, 1)

                cols = st.columns(num, gap="medium")
                for i in range(num):
                    with cols[i]:
                        st.image(imgs[i], caption=f"Sample {i+1}", use_container_width=True)

                        if classifier is not None:
                            resized = tf.image.resize(imgs[i], [224, 224])
                            pred = classifier.predict(np.expand_dims(resized, 0), verbose=0)
                            cidx = int(np.argmax(pred))
                            st.caption(f"{CLASS_EMOJI[cidx]} **{CLASSES[cidx]}** · {np.max(pred)*100:.0f}%")

                        # Download button
                        st.download_button(
                            f"⬇️ Download",
                            data=array_to_png_bytes(imgs[i]),
                            file_name=f"synthetic_ecg_{i+1}.png",
                            mime="image/png",
                            key=f"dl_{i}"
                        )

# ═══════════════════════════════════════════════════════════════
#  PAGE 4: RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════
elif "Anomaly Detection" in page:
    st.title("Autoencoder Reconstruction")
    st.markdown("Compress and reconstruct ECG images to measure reconstruction error. High error signals potential anomalies.")
    st.markdown("---")

    autoencoder = load_model(AUTOENCODER_PATH)

    if autoencoder is None:
        st.error("Autoencoder model not found. Place `autoencoder.keras` in the app directory.")
    else:
        uploaded = st.file_uploader("Upload a CWT spectrogram image", type=["png", "jpg", "jpeg"], key="ae")

        if uploaded:
            image = Image.open(uploaded)
            img = image.resize((64, 64)).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            inp = np.expand_dims(arr, 0)

            if st.button("Reconstruct"):
                with st.spinner("Reconstructing..."):
                    recon = autoencoder.predict(inp, verbose=0)[0]
                    recon = np.clip(recon, 0, 1)
                    mse = float(np.mean((arr - recon) ** 2))

                    col1, col2 = st.columns(2, gap="large")
                    with col1:
                        st.image(arr, caption="Original", use_container_width=True)
                    with col2:
                        st.image(recon, caption="Reconstructed", use_container_width=True)

                    st.metric("Reconstruction Error (MSE)", f"{mse:.6f}")

                    if mse > 0.05:
                        st.warning(" High reconstruction error — possible anomalous pattern detected.")
                    else:
                        st.success(" Low reconstruction error — pattern is consistent with trained distributions.")

# ═══════════════════════════════════════════════════════════════
#  PAGE 5: ABOUT
# ═══════════════════════════════════════════════════════════════
elif "About" in page:
    st.title("About CardioVision AI")
    st.markdown("---")

    # ── Hero stats row ──
    c1, c2, c3 = st.columns(3)
    c1.metric("Deployed Models", "3")
    c2.metric("Arrhythmia Classes", "5")
    c3.metric("Dataset Size", "109,446")

    st.markdown("---")

    # ── Project Overview ──
    st.markdown("## Project Overview")
    st.markdown("""
    This application is an end-to-end deep learning pipeline for analyzing electrocardiogram (ECG) data. 
    It processes 1D temporal signals into Continuous Wavelet Transform (CWT) spectrograms to perform 
    image-based arrhythmia classification via transfer learning architectures.

    The system is trained entirely on the MIT-BIH Arrhythmia Database, mapping patient signals into 
    5 distinct AAMI diagnostic categories.
    """)

    st.markdown("---")

    # ── Data Pipeline ──
    st.markdown("## Data Pipeline")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 1. Raw Signal
        - MIT-BIH PhysioNet DB
        - 187 timesteps
        - 109,446 samples
        - 5 primary classes
        """)
    with col2:
        st.markdown("""
        #### 2. CWT Transform
        - Mexican Hat wavelet
        - 127 frequency scales
        - 2D time-frequency mapping
        - Viridis encoding
        """)
    with col3:
        st.markdown("""
        #### 3. Classification
        - ResNet-50 backbone
        - 224×224 RGB tensor
        - Softmax mapping
        - Streamlit endpoint
        """)

    st.markdown("---")

    # ── Deployed Models ──
    st.markdown("## Deployed Architecture")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.info("**Arrhythmia Classifier**")
        st.markdown("""
        **Type**: ResNet-50 (Fine-tuned)
        **Input**: 224x224 CWT Spectrogram
        **Output**: 5-class Distribution
        """)
        
    with col_b:
        st.success("**ECG Generator**")
        st.markdown("""
        **Type**: Deep Convolutional GAN
        **Input**: 128-dim Latent Vector
        **Output**: 64x64 CWT Spectrogram
        """)

    with col_c:
        st.warning("**Anomaly Detector**")
        st.markdown("""
        **Type**: Conv. Autoencoder
        **Domain**: 64x64 Representation
        **Metric**: Mean Squared Error
        """)

    st.markdown("---")
    st.markdown("## Class Distribution (MIT-BIH)")
    class_data = pd.DataFrame({
        'Class': ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)'],
        'Train Samples': [72471, 2223, 5788, 641, 6431],
        'Color': ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    })

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(class_data['Class'], class_data['Train Samples'], color=class_data['Color'], edgecolor='none', height=0.6)
    ax.set_xlabel('Number of Samples', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, class_data['Train Samples']):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ── Deployment Features ──
    st.markdown("##  Deployment Features")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        ####  Classification
        Upload CWT images or raw CSV signals 
        for instant arrhythmia prediction with 
        confidence scores.
        """)
        st.markdown("""
        ####  Generation
        Synthesize new ECG spectrograms 
        using DCGAN with auto-classification.
        """)
    with f2:
        st.markdown("""
        ####  Batch Prediction
        Classify hundreds of heartbeats at once 
        with bar/pie chart distribution 
        and CSV export.
        """)
        st.markdown("""
        ####  Reconstruction
        Autoencoder-based anomaly detection 
        through reconstruction MSE analysis.
        """)
    with f3:
        st.markdown("""
        ####  Downloads
        Export generated synthetic images 
        as PNG and batch results as CSV.
        """)
        st.markdown("""
        ####  Sample Data
        Built-in demo signal for instant 
        testing without file uploads.
        """)

    st.markdown("---")

    # ── Tech Stack ──
    st.markdown("## 🛠️ Tech Stack")
    t1, t2, t3, t4 = st.columns(4)
    t1.markdown("**Deep Learning**\n\nTensorFlow · Keras")
    t2.markdown("**Signal Processing**\n\nPyWavelets · OpenCV")
    t3.markdown("**Visualization**\n\nMatplotlib · Streamlit")
    t4.markdown("**Data Science**\n\nPandas · Scikit-learn")

    st.markdown("---")
    st.markdown("*Built as part of the **24AI636 Deep Learning Mini-Project** · MTech AI, 2026*")

