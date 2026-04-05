#  CardioVision AI — ECG Arrhythmia Detection & Synthesis

Deep Learning pipeline for ECG arrhythmia classification, synthetic generation, and anomaly detection. 
Built using the MIT-BIH Arrhythmia Database, this project transforms 1D ECG signals into 2D Continuous Wavelet Transform (CWT) spectrograms for advanced image-based analysis.

##  Live Demo
Access the deployed Streamlit dashboard here: ** https://arrhythmia-classification-8dkxe7aq4sn8edpfmzqi8x.streamlit.app/ **

---

##  Project Overview
Cardiac arrhythmias are irregular heartbeat patterns that can lead to serious health complications. This project applies various deep learning architectures to automatically classify ECG signals into 5 continuous arrhythmia categories defined by the AAMI standard.

### Dataset Features (MIT-BIH)
- **Signal**: 187 timesteps per heartbeat
- **Volume**: 109,446 total samples
- **Classes**: 
  - 🟢 Normal (N)
  - 🔵 Supraventricular (S)
  - 🔴 Ventricular (V)
  - 🟠 Fusion (F)
  - 🟣 Unknown (Q)

---

##  Model Evolution & Architecture

Throughout the development of this project, several models were tested and evaluated to find the best approach for signal analysis.

### Review 1: Baseline Models (1D Signals)
Initial experiments focused on direct classification of the raw 1D temporal signals.
- **Multi-Layer Perceptron (MLP)**: Baseline network `Dense(128) → Dense(64) → Dense(5)`. Provided foundational accuracy but struggled with complex temporal variations.
- **1D Convolutional Neural Network (1D CNN)**: Utilized `Conv1D` and `MaxPool` layers. Showed significant improvement in capturing local temporal features compared to the MLP.

### Review 2: Advanced Classification (Time-Series vs. Time-Frequency)
Explored sophisticated architectures, introducing the conversion of 1D signals to 2D CWT spectrograms.
- **ResNet-50 (Fine-tuned)**: Transitioned from 1D signals to 2D CWT images (224x224). Leveraged transfer learning on an ImageNet pre-trained ResNet-50 backbone. **This approach yielded the highest classification accuracy.**
- **LSTM + Attention Layer**: Bidirectional LSTM focusing on sequential dependencies within the 1D signal, augmented with an attention mechanism to weigh critical timesteps.
- **GRU / BiLSTM**: Efficient recurrent architectures for direct temporal sequence analysis.

### Review 3: Generative AI & Anomaly Detection
Expanded the scope beyond basic classification to include synthetic data generation and unsupervised anomaly detection.
- **Deep Convolutional GAN (DCGAN)**: 
  - *Generator*: Maps a 128-dim latent vector to a 64x64 CWT spectrogram.
  - *Discriminator*: Distinguishes real CWT images from synthetic ones.
  - *Purpose*: Generates synthetic ECG patterns for data augmentation and analysis.
- **Convolutional Autoencoder**: 
  - *Architecture*: Symmetrical Encoder-Decoder compressing 64x64 images to a dense latent representation.
  - *Purpose*: Reconstructs input images. High Mean Squared Error (MSE) between input and reconstruction indicates potential anomalous patterns not present in the training distribution.

---

##  Tech Stack
- **Deep Learning**: TensorFlow, Keras
- **Signal Processing**: PyWavelets, OpenCV
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Dashboard Deployment**: Streamlit, Matplotlib

---

##  Local Setup & Deployment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sidharth1snair/arrhythmia-classification.git
   cd arrhythmia-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```
   *(Note: The required `.keras` model weights will automatically download from Google Drive on the first launch)*

---
*Built as part of the 24AI636 Deep Learning Mini-Project (MTech AI, 2026)*
