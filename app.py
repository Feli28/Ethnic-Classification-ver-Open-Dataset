
import os
import pickle
import threading
from collections import deque

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from PIL import Image
from facenet_pytorch import MTCNN
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
)

from feature_extraction import (
    detect_landmarks,
    extract_antropometri22,
    extract_lbp,
)


# =========================================================
# KONFIGURASI
# =========================================================

st.set_page_config(
    page_title="Klasifikasi Etnis Wajah",
    page_icon="👤",
    layout="wide",
)

MODEL_DIR = "models"
IMAGE_SIZE = (510, 510)

ANTRO_MODEL_PATH = os.path.join(
    MODEL_DIR, "antro_own_fair_no_bg_svm.pkl"
)
ANTRO_SCALER_PATH = os.path.join(
    MODEL_DIR, "antro_own_fair_no_bg_scaler.pkl"
)
LBP_MODEL_PATH = os.path.join(
    MODEL_DIR, "lbp_own_fair_svm.pkl"
)
LBP_SCALER_PATH = os.path.join(
    MODEL_DIR, "lbp_own_fair_scaler.pkl"
)


# =========================================================
# LOAD MODEL DAN MTCNN
# =========================================================

@st.cache_resource
def load_models():
    paths = [
        ANTRO_MODEL_PATH,
        ANTRO_SCALER_PATH,
        LBP_MODEL_PATH,
        LBP_SCALER_PATH,
    ]

    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"File tidak ditemukan: {path}"
            )

    with open(ANTRO_MODEL_PATH, "rb") as f:
        antro_model = pickle.load(f)

    with open(ANTRO_SCALER_PATH, "rb") as f:
        antro_scaler = pickle.load(f)

    with open(LBP_MODEL_PATH, "rb") as f:
        lbp_model = pickle.load(f)

    with open(LBP_SCALER_PATH, "rb") as f:
        lbp_scaler = pickle.load(f)

    return antro_model, antro_scaler, lbp_model, lbp_scaler


@st.cache_resource
def load_mtcnn():
    # Samakan parameter MTCNN dengan preprocessing training.
    return MTCNN(
        keep_all=False,
        device="cpu",
    )


# =========================================================
# PREPROCESSING MTCNN
# =========================================================

def crop_face_mtcnn(image_pil, mtcnn):
    """
    Mengikuti aturan crop yang diberikan:
    confidence > 0.95,
    pusat crop bergeser ke bawah 5% tinggi box,
    sisi crop = 95% dari max(lebar, tinggi),
    resize 510x510 dengan LANCZOS.
    """
    image_pil = image_pil.convert("RGB")
    img_w, img_h = image_pil.size

    boxes, probs = mtcnn.detect(image_pil)

    if boxes is None or probs is None or len(boxes) == 0:
        return None, "Wajah tidak terdeteksi oleh MTCNN."

    if probs[0] is None or probs[0] <= 0.95:
        return None, "Confidence deteksi wajah harus lebih dari 0.95."

    x1, y1, x2, y2 = boxes[0]

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return None, "Bounding box wajah tidak valid."

    cx = x1 + w / 2
    cy = y1 + h / 2
    cy_adjusted = cy + h * 0.05

    tight_side = max(w, h) * 0.95
    half_side = tight_side / 2

    new_x1 = max(0, int(cx - half_side))
    new_y1 = max(0, int(cy_adjusted - half_side))
    new_x2 = min(img_w, int(cx + half_side))
    new_y2 = min(img_h, int(cy_adjusted + half_side))

    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return None, "Hasil crop wajah tidak valid."

    cropped = image_pil.crop(
        (new_x1, new_y1, new_x2, new_y2)
    )

    resized = cropped.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS,
    )

    return resized, None


# =========================================================
# PREDIKSI ANTROPOMETRI
# =========================================================

def predict_antro(image_bgr, model, scaler):
    landmarks_list = detect_landmarks(image_bgr)

    if not landmarks_list:
        raise ValueError(
            "MediaPipe Tasks gagal mendeteksi landmark wajah."
        )

    landmarks = landmarks_list[0]
    features = np.asarray(
        extract_antropometri22(landmarks),
        dtype=np.float32,
    ).reshape(1, -1)

    if features.shape[1] != 22:
        raise ValueError(
            f"Model antropometri membutuhkan 22 fitur, "
            f"tetapi ditemukan {features.shape[1]}."
        )

    X_scaled = scaler.transform(features)
    prediction = model.predict(X_scaled)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = float(np.max(probabilities))

    return prediction, confidence, features.ravel()


# =========================================================
# PREDIKSI LBP
# =========================================================

def predict_lbp(image_bgr, model, scaler):
    features = np.asarray(
        extract_lbp(image_bgr),
        dtype=np.float32,
    ).reshape(1, -1)

    if features.shape[1] != 2304:
        raise ValueError(
            f"Model LBP membutuhkan 2304 fitur, "
            f"tetapi ditemukan {features.shape[1]}."
        )

    X_scaled = scaler.transform(features)
    prediction = model.predict(X_scaled)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = float(np.max(probabilities))

    return prediction, confidence, features.ravel()


# =========================================================
# LIVE WEBCAM
# =========================================================

class WebcamProcessor(VideoProcessorBase):
    """
    Menyimpan frame webcam terbaru agar bisa digunakan
    ketika pengguna menekan tombol Prediksi.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frames = deque(maxlen=1)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.latest_frame = image.copy()

        return frame

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()


# =========================================================
# FUNGSI TAMPILKAN HASIL
# =========================================================

def process_image(image_pil, method, models, mtcnn):
    antro_model, antro_scaler, lbp_model, lbp_scaler = models

    cropped_pil, error = crop_face_mtcnn(image_pil, mtcnn)

    if error:
        st.error(error)
        return

    st.image(
        cropped_pil,
        caption="Hasil crop MTCNN (510 × 510)",
        width=350,
    )

    image_bgr = cv2.cvtColor(
        np.asarray(cropped_pil),
        cv2.COLOR_RGB2BGR,
    )

    if method == "Antropometri 22 Fitur":
        prediction, confidence, features = predict_antro(
            image_bgr,
            antro_model,
            antro_scaler,
        )
    else:
        prediction, confidence, features = predict_lbp(
            image_bgr,
            lbp_model,
            lbp_scaler,
        )

    st.success(f"Hasil prediksi: {prediction}")
    st.write(f"Metode: {method}")
    st.write(f"Jumlah fitur: {len(features)}")

    if confidence is not None:
        st.write(
            f"Probabilitas kelas tertinggi: "
            f"{confidence * 100:.2f}%"
        )
        st.caption(
            "Nilai probabilitas SVM bukan jaminan bahwa prediksi benar."
        )

    with st.expander("Lihat nilai fitur"):
        st.write(features)


# =========================================================
# ANTARMUKA
# =========================================================

st.title("Klasifikasi Etnis Wajah")
st.write(
    "Sistem klasifikasi menggunakan fitur antropometri "
    "atau Local Binary Pattern (LBP)."
)

method = st.radio(
    "Pilih metode ekstraksi fitur",
    ["Antropometri 22 Fitur", "LBP 2304 Fitur"],
    horizontal=True,
)

source = st.radio(
    "Pilih sumber gambar",
    ["Upload Gambar", "Ambil Foto", "Live Webcam"],
    horizontal=True,
)

# Load resources saat dibutuhkan.
try:
    models = load_models()
    mtcnn = load_mtcnn()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# ---------------------------------------------------------
# UPLOAD GAMBAR
# ---------------------------------------------------------

if source == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Pilih gambar wajah",
        type=["jpg", "jpeg", "png"],
        key="upload_image",
    )

    if uploaded_file is not None:
        try:
            image_pil = Image.open(uploaded_file).convert("RGB")

            st.image(
                image_pil,
                caption="Gambar asli",
                use_container_width=True,
            )

            if st.button(
                "Prediksi",
                type="primary",
                key="predict_upload",
            ):
                with st.spinner("Memproses gambar..."):
                    process_image(
                        image_pil, method, models, mtcnn
                    )

        except Exception as e:
            st.error(f"Gagal membaca gambar: {e}")


# ---------------------------------------------------------
# AMBIL FOTO LANGSUNG
# ---------------------------------------------------------

elif source == "Ambil Foto":
    camera_image = st.camera_input(
        "Ambil foto menggunakan kamera",
        key="capture_image",
    )

    if camera_image is not None:
        try:
            image_pil = Image.open(camera_image).convert("RGB")

            if st.button(
                "Prediksi",
                type="primary",
                key="predict_capture",
            ):
                with st.spinner("Memproses foto..."):
                    process_image(
                        image_pil, method, models, mtcnn
                    )

        except Exception as e:
            st.error(f"Gagal memproses foto: {e}")


# ---------------------------------------------------------
# LIVE WEBCAM
# ---------------------------------------------------------

elif source == "Live Webcam":
    st.info(
        "Izinkan akses kamera pada browser. "
        "Tunggu sampai video muncul, lalu tekan Prediksi "
        "untuk memproses frame terbaru."
    )

    ctx = webrtc_streamer(
        key="live-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=WebcamProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.state.playing and ctx.video_processor:
        if st.button(
            "Prediksi Frame Webcam",
            type="primary",
            key="predict_webcam",
        ):
            frame_bgr = ctx.video_processor.get_latest_frame()

            if frame_bgr is None:
                st.warning(
                    "Frame belum tersedia. Pastikan webcam aktif."
                )
            else:
                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )
                image_pil = Image.fromarray(frame_rgb)

                st.image(
                    image_pil,
                    caption="Frame yang akan diproses",
                    use_container_width=True,
                )

                with st.spinner("Memproses frame webcam..."):
                    process_image(
                        image_pil, method, models, mtcnn
                    )
    else:
        st.caption("Webcam belum aktif.")
