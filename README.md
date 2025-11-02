# SHIELD-AI: Phishing URL & Deepfake Detector

A Streamlit app that combines:
- Phishing URL classification (scikit-learn + TF-IDF features)
- Deepfake video detection (Keras model over sampled frames with optional face-crop and TTA)

## Features
- Upload a video and get a REAL/FAKE decision with supporting frame scores
- URL check with probabilities (Phishing vs Legitimate)
- Lazy imports for heavy deps so the phishing checker still works if deepfake deps are missing
- Threshold control and advanced settings (frame stride, max frames, face margin, TTA)

## Repository layout
```
app.py                            # Streamlit app entry
requirements.txt                  # Pinned runtime dependencies
phishing_rf_model.pkl             # Trained phishing classifier
phishing_tfidf.pkl                # TF-IDF vectorizer for phishing
deepfake_detector_mobilenet_subset.keras  # Deepfake model used by the app
image.png, deepfake.jpg           # UI images
threshold.txt                     # (optional) default decision threshold
```

## Quick start (Windows, Python 3.11)
1) Create a virtual environment and activate it
```powershell
python -m venv venv_3.11
./venv_3.11/Scripts/Activate.ps1
```

2) Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

3) Run the app
```powershell
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Troubleshooting
- Error: ModuleNotFoundError: No module named 'numpy._core' or "numpy.core.multiarray failed to import"
  - Cause: ABI mismatch between NumPy, OpenCV, and/or TensorFlow.
  - Fix: We pin compatible versions in requirements.txt. If the error appears, force-reinstall:
  ```powershell
  pip install --no-cache-dir --force-reinstall numpy==1.23.5 opencv-python-headless==4.8.0.74 tensorflow==2.15.0 keras==2.15.0
  ```
- OpenCV video read issues:
  - Ensure the uploaded video is a supported format (mp4, avi, mov). Try lowering frame stride in Advanced Settings.
- Model not found message:
  - The app tries these in order: deepfake_detector_mobilenet_subset.keras → deepfake_detector_simple_subset.keras → deepfake_detector_simple.keras → deepfake_detector_model.keras → deepfake_detector_model.h5. Keep exactly one to avoid confusion.


