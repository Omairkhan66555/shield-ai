import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
import tldextract
import tempfile
import base64
from pathlib import Path

# --------- Card Helper Function -------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
    
        data = f.read()
    return base64.b64encode(data).decode()

def card(image_path, emoji, title, description, button_color, key=None):
    card_html = f"""
    <div style="
        border-radius: 18px;
        box-shadow: 0 2px 24px rgba(10,255,60,0.07), 0 0px 2px 0px #111;
        background: #191e20;
        margin: 18px 0 28px 0;
        padding: 0;
        max-width: 500px;
    ">
        <img src="data:image/jpg;base64,{get_base64_of_bin_file(image_path)}"
            style="width: 100%; height: 160px; object-fit: cover; border-radius: 16px 16px 0 0;"/>
        <div style="padding: 20px;">
            <span style="font-size: 1.25rem;">{emoji} <span style="font-weight: 700">{title}</span></span>
            <div style="margin: 8px 0 18px 0; color: #aaa; font-size: 1.05rem;">{description}</div>
        </div>
    </div>
    <style>
    div[data-testid="stButton"] button#{key} {{
        background: {button_color} !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.55rem !important;
        font-size: 1.06rem !important;
        font-weight: bold !important;
        margin-left: 20px !important;
        margin-top: -36px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 3px 10px #111a;
        transition: filter 0.2s;
    }}
    div[data-testid="stButton"] button#{key}:hover {{
        filter: brightness(1.2);
    }}
    </style>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    return st.button("Go", key=key)

# -------- CSS (hacker-theme) ----------
st.markdown("""
<style>
body {
  background: #111;
  color: #33ff33;
}
.stApp {
  background-color: #141d1d;
  color: #33ff33;
  font-family: 'Fira Mono', monospace;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------- App Title ----------
st.markdown(
    "<h1 style='text-align:center; color:#33ff33; margin:18px 0 10px 0;'><b>🛡️ SHIELD-AI:Phishing URL & DeepFake Detector</b></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#17ff17; font-size: 1.13rem;'>#Safe-Smart-Shielded</p>",
    unsafe_allow_html=True,
)
st.markdown('<hr style="border:1px solid #444;">', unsafe_allow_html=True)

# --------- CARD UI --------------
colA, colB = st.columns(2)

# Phishing model load
phishing_model = joblib.load('phishing_rf_model.pkl')
vectorizer = joblib.load('phishing_tfidf.pkl')

def extract_features(url):
    feats = {}
    feats['url_length'] = len(url)
    feats['num_digits'] = sum(c.isdigit() for c in url)
    feats['num_dots'] = url.count('.')
    feats['num_hyphens'] = url.count('-')
    feats['num_at'] = url.count('@')
    feats['has_https'] = float(url.lower().startswith('https'))
    ext = tldextract.extract(url)
    feats['domain_length'] = len(ext.domain)
    return feats

def predict_phishing(url_input):
    feats = pd.DataFrame([extract_features(url_input)])
    tfidf_vec = vectorizer.transform([url_input])
    final_vec = hstack([tfidf_vec, feats])
    pred_label = phishing_model.predict(final_vec)[0]
    phishing_index = list(phishing_model.classes_).index('phishing')
    prob = phishing_model.predict_proba(final_vec)[0][phishing_index]
    label = 'Phishing' if pred_label == 'phishing' else 'Legitimate'
    return label, prob

with colA:
    if card(
        "image.png", "🔗", "Phishing URL Check",
        "Check if a website link is suspicious or safe in one click.",
        "#2A7CF7", key="phishing"
    ):
        st.session_state.section = "phishing"

with colB:
    if card(
        "deepfake.jpg", "🎭", "Deepfake Video Check",
        "Detect AI-generated fake videos confidently and instantly.",
        "#38B64A", key="deepfake"
    ):
        st.session_state.section = "deepfake"

# Functional Sections

if st.session_state.get("section") == "phishing":
    st.subheader("🔗 Phishing URL Detector")
    url = st.text_input("Enter URL to check:")
    if st.button("Check URL", key="button_phishing"):
        if url:
            label, confidence = predict_phishing(url)
            if label == "Phishing":
                st.markdown(
                    f"""<div style='background:#1d2b1d; color:#ff4444; padding:18px; border-radius:8px;
                        font-size:1.3rem; font-family:Fira Mono,monospace; border:2.5px solid #ff1010; text-align:center;'>
                        <b>🚨 URL IS PHISHING!</b><br>
                        <span style='color:#fff;'>Confidence: <b>{confidence:.2f}</b></span>
                    </div>""",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""<div style='background:#161d14; color:#33ff33; padding:18px; border-radius:8px;
                        font-size:1.3rem; font-family:Fira Mono,monospace; border:2.5px solid #16ff16; text-align:center;'>
                        <b>✅ LEGITIMATE URL</b><br>
                        <span style='color:#fff;'>Confidence: <b>{confidence:.2f}</b></span>
                    </div>""",
                    unsafe_allow_html=True)
        else:
            st.warning("Please enter a URL.")

if st.session_state.get("section") == "deepfake":
    st.subheader("🎭 Deepfake Video Detector")
    uploaded_video = st.file_uploader("Upload video for deepfake check", type=["mp4", "avi", "mov"])
    
    # -------- Deepfake helpers (MobileNet subset model, same as notebook) --------
    @st.cache_resource(show_spinner=False)
    def _load_any_model_streamlit():
        """Load preferred model artifact from disk (cached)."""
        # Lazy import TensorFlow only when needed
        from tensorflow.keras.models import load_model
        candidates = [
            # Preferred newer artifacts
            Path('deepfake_detector_mobilenet_subset.keras'),
        ]
        for c in candidates:
            if c.exists():
                st.write(f"Loading model: {c}")
                return load_model(str(c))
        raise FileNotFoundError("No supported model artifacts found. Train first.")

    def _detect_and_crop_face_img(bgr_img, margin=0.2):
        import cv2  # lazy import
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        cascade_path = str(Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if faces is None or len(faces) == 0:
            return bgr_img
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        mh = int(h * margin)
        mw = int(w * margin)
        x0 = max(0, x - mw)
        y0 = max(0, y - mh)
        x1 = min(bgr_img.shape[1], x + w + mw)
        y1 = min(bgr_img.shape[0], y + h + mh)
        return bgr_img[y0:y1, x0:x1]

    def _predict_prob_frame(model, bgr_img, autoface=True, margin=0.2, tta=True):
        import cv2  # lazy import
        try:
            h, w = int(model.input_shape[1]), int(model.input_shape[2])
        except Exception:
            h, w = 160, 160
        img = bgr_img.copy()
        if autoface:
            img = _detect_and_crop_face_img(img, margin=margin)
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        x = np.expand_dims(img, axis=0)
        p1 = float(model.predict(x, verbose=0)[0][0])
        if not tta:
            return p1
        img2 = cv2.flip(img, 1)
        x2 = np.expand_dims(img2, axis=0)
        p2 = float(model.predict(x2, verbose=0)[0][0])
        return (p1 + p2) / 2.0

    def _process_video_streamlit(video_path: str, frame_stride=10, max_frames=120, autoface=True, margin=0.2, tta=True, results_dir='streamlit_results'):
        import cv2  # lazy import
        if not Path(video_path).exists():
            raise FileNotFoundError(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError('Could not open video: ' + video_path)
        model = _load_any_model_streamlit()
        probs = []
        details = []  # (fname, score, info)
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        frames_collected = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % max(1, frame_stride) != 0:
                frame_idx += 1
                continue
            prob = _predict_prob_frame(model, frame, autoface=autoface, margin=margin, tta=tta)
            probs.append(prob)
            # Save thumbnail for UI and record detail
            thumb = cv2.resize(frame, (320, 180))
            fname = f"frame_{frame_idx:06d}.jpg"
            out_path = Path(results_dir) / f"{prob:.4f}_".replace('.', '_')  # avoid issues with '.' in Streamlit static serving
            # Keep original expected pattern: {score:.4f}_{fname}
            out_path = Path(results_dir) / f"{prob:.4f}_{fname}"
            cv2.imwrite(str(out_path), cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
            details.append((fname, prob, f'idx={frame_idx}'))
            frames_collected += 1
            frame_idx += 1
            if frames_collected >= max_frames:
                break
        cap.release()
        return np.array(probs, dtype=np.float32), details

    def _read_default_threshold():
        p = Path('threshold.txt')
        if p.exists():
            try:
                return float(p.read_text().strip())
            except Exception:
                return 0.45
        return 0.45

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_video.read())
            tmp_file_path = tmp_file.name
            # Advanced settings in an expander (aligned with notebook evaluator)
            with st.expander("Advanced Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    threshold = st.slider("Decision threshold", min_value=0.1, max_value=0.9, value=float(_read_default_threshold()), step=0.01)
                    frame_stride = st.number_input("Frame stride", min_value=1, max_value=60, value=10, step=1)
                    max_frames = st.number_input("Max frames", min_value=10, max_value=500, value=150, step=10)
                with col2:
                    autoface = st.checkbox("Auto face-crop", value=True)
                    tta = st.checkbox("Enable TTA (flip)", value=True)
                    face_margin = st.slider("Face margin", min_value=0.0, max_value=0.5, value=0.2, step=0.05)

            with st.spinner("Analyzing video..."):
                try:
                    probs, details = _process_video_streamlit(
                        tmp_file_path,
                        frame_stride=int(frame_stride),
                        max_frames=int(max_frames),
                        autoface=bool(autoface),
                        margin=float(face_margin),
                        tta=bool(tta),
                        results_dir='streamlit_results'
                    )
                except ImportError as e:
                    st.error(
                        "Dependency import failed (likely NumPy/OpenCV/TensorFlow mismatch). "
                        "Please reinstall compatible versions in your venv: NumPy < 2 (e.g., 1.23.5) and OpenCV 4.8.0.74.\n"
                        "Steps: close Streamlit, then run: pip install --force-reinstall --no-cache-dir numpy==1.23.5 opencv-python-headless==4.8.0.74"
                    )
                    st.stop()
            if len(probs) == 0:
                st.warning('No frames processed. Try lowering frame stride or check the video file.')
                st.stop()

            # Aggregate decision: mean fake_prob vs threshold
            mean_prob = float(probs.mean())
            label = "REAL" if mean_prob >= float(threshold) else "FAKE"
            confidence = mean_prob

            # Show analysis details
            st.markdown("### Analysis Details")
            col1, col2 = st.columns(2)

            with col1:
                total_frames = len(details)
                st.metric("Frames evaluated", total_frames)
                st.metric("Frames >= threshold", int((probs >= float(threshold)).sum()))

            with col2:
                st.metric("Mean real_prob", f"{mean_prob:.3f}")
                st.metric("Threshold", f"{float(threshold):.3f}")

            # Show face thumbnails and scores
            if len(details) > 0:
                with st.expander("View Frame Scores", expanded=False):
                    st.markdown("#### Frame Analysis")
                    for fname, score, info in details[:30]:  # cap at 30 rows for UI
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            try:
                                img_path = f"streamlit_results/{score:.4f}_{fname}"
                                st.image(img_path, width=120)
                            except Exception:
                                st.write("Image not available")
                        with c2:
                            st.write(f"Score: {score:.3f} | {info}")
        if label == "FAKE":
            st.markdown(
                f"""<div style='background:#1d1d14; color:#ff4444; padding:18px; border-radius:8px;
                    font-size:1.3rem; font-family:Fira Mono,monospace; border:2.5px solid #ff1010; text-align:center;'>
                    <b>🚨 VIDEO IS DEEPFAKE!</b><br>
                    <span style='color:#fff;'>Confidence: <b>{confidence:.2f}</b></span>
                </div>""",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div style='background:#141614; color:#33ff33; padding:18px; border-radius:8px;
                    font-size:1.3rem; font-family:Fira Mono,monospace; border:2.5px solid #16ff16; text-align:center;'>
                    <b>✅ VIDEO IS REAL</b><br>
                    <span style='color:#fff;'>Confidence: <b>{confidence:.2f}</b></span>
                </div>""",
                unsafe_allow_html=True)

st.markdown("<hr><p style='text-align:center; color:#879; font-size:.9rem;'>Designed by <b>Shield AI 🛡️</b> | <span style='color:#33ff33;'>Hacker Mode</span></p>", unsafe_allow_html=True)
