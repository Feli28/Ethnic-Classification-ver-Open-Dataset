import os
import joblib
import threading
import hashlib

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from PIL import Image, ImageOps
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

        if os.path.getsize(path) == 0:
            raise ValueError(
                f"File model kosong: {path}"
            )

    return (
        joblib.load(ANTRO_MODEL_PATH),
        joblib.load(ANTRO_SCALER_PATH),
        joblib.load(LBP_MODEL_PATH),
        joblib.load(LBP_SCALER_PATH),
    )


@st.cache_resource
def load_mtcnn():
    return MTCNN(
        keep_all=False,
        device="cpu",
    )


# =========================================================
# PREPROCESSING MTCNN
# =========================================================

def crop_face_mtcnn(image_pil, mtcnn):
    image_pil = image_pil.convert("RGB")
    img_w, img_h = image_pil.size

    boxes, probs = mtcnn.detect(image_pil)

    if boxes is None or probs is None or len(boxes) == 0:
        return None, "Wajah tidak terdeteksi oleh MTCNN."

    if probs[0] is None or probs[0] <= 0.95:
        return None, (
            "Confidence deteksi wajah harus lebih dari 0.95."
        )

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
# TAMPILKAN HASIL PREDIKSI
# =========================================================

```python
def show_prediction(image_pil, method, models, mtcnn):
    antro_model, antro_scaler, lbp_model, lbp_scaler = models

    # Input sudah di-flip sebelum masuk ke fungsi ini.
    cropped_pil, error = crop_face_mtcnn(image_pil, mtcnn)

    if error:
        st.error(error)
        return

    # Tampilkan hanya hasil crop MTCNN.
    st.image(
        cropped_pil,
        caption="Input wajah (MTCNN, 510 × 510)",
        width=350,
    )

    image_bgr = cv2.cvtColor(
        np.asarray(cropped_pil),
        cv2.COLOR_RGB2BGR,
    )

    if method == "Antropometri 22 Fitur":
        prediction, _, _ = predict_antro(
            image_bgr,
            antro_model,
            antro_scaler,
        )
    else:
        prediction, _, _ = predict_lbp(
            image_bgr,
            lbp_model,
            lbp_scaler,
        )

    st.success(f"Hasil prediksi: {prediction}")
```
# =========================================================
# LIVE WEBCAM
# =========================================================

class WebcamProcessor(VideoProcessorBase):
    def __init__(self, models, mtcnn, method):
        self.lock = threading.Lock()
        self.latest_frame = None

        self.models = models
        self.mtcnn = mtcnn
        self.method = method

        self.frame_count = 0
        self.last_box = None
        self.last_prediction = None
        self.last_error = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Flip horizontal untuk tampilan dan prediksi.
        image = cv2.flip(image, 1)

        self.frame_count += 1

        # Jalankan deteksi dan prediksi setiap 10 frame.
        # Frame lain tetap menampilkan hasil terakhir.
        if self.frame_count % 10 == 1:
            try:
                image_rgb = cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB
                )
                image_pil = Image.fromarray(image_rgb)

                boxes, probs = self.mtcnn.detect(image_pil)

                if (
                    boxes is not None
                    and probs is not None
                    and len(boxes) > 0
                    and probs[0] is not None
                    and probs[0] > 0.95
                ):
                    x1, y1, x2, y2 = boxes[0]
                    self.last_box = (
                        int(x1), int(y1), int(x2), int(y2)
                    )

                    # Crop dan resize mengikuti preprocessing.
                    cropped, error = crop_face_mtcnn(
                        image_pil, self.mtcnn
                    )

                    if error:
                        self.last_prediction = None
                        self.last_error = error
                    else:
                        crop_bgr = cv2.cvtColor(
                            np.asarray(cropped),
                            cv2.COLOR_RGB2BGR,
                        )

                        antro_model, antro_scaler, lbp_model, lbp_scaler = (
                            self.models
                        )

                        if self.method == "Antropometri 22 Fitur":
                            prediction, _, _ = predict_antro(
                                crop_bgr,
                                antro_model,
                                antro_scaler,
                            )
                        else:
                            prediction, _, _ = predict_lbp(
                                crop_bgr,
                                lbp_model,
                                lbp_scaler,
                            )

                        self.last_prediction = str(prediction)
                        self.last_error = None
                else:
                    self.last_box = None
                    self.last_prediction = None
                    self.last_error = "Wajah belum terdeteksi"

            except Exception as e:
                self.last_error = str(e)

        # Gambar bounding box pada frame video.
        if self.last_box is not None:
            x1, y1, x2, y2 = self.last_box

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"Prediksi: {self.last_prediction}"
                if self.last_prediction
                else "Memproses wajah..."
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.65
            thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, scale, thickness
            )

            label_y = max(y1 - 10, text_h + 10)

            cv2.rectangle(
                image,
                (x1, label_y - text_h - 8),
                (x1 + text_w + 8, label_y + baseline),
                (0, 180, 0),
                -1,
            )

            cv2.putText(
                image,
                label,
                (x1 + 4, label_y - 4),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        elif self.last_error:
            cv2.putText(
                image,
                self.last_error[:60],
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        with self.lock:
            self.latest_frame = image.copy()

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

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

try:
    models = load_models()
    mtcnn = load_mtcnn()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# =========================================================
# UPLOAD GAMBAR — TANPA FLIP
# =========================================================

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
                    show_prediction(
                        image_pil,
                        method,
                        models,
                        mtcnn,
                    )

        except Exception as e:
            st.error(f"Gagal membaca gambar: {e}")


# =========================================================
# AMBIL FOTO — FLIP DAN PREDIKSI OTOMATIS
# =========================================================

elif source == "Ambil Foto":
    camera_image = st.camera_input(
        "Ambil foto menggunakan kamera",
        key="capture_image",
    )

    if camera_image is not None:
        try:
            image_pil = Image.open(camera_image).convert("RGB")

            # Flip horizontal sebelum crop MTCNN.
            image_pil = ImageOps.mirror(image_pil)

            image_hash = hashlib.md5(
                camera_image.getvalue()
            ).hexdigest()

            prediction_key = f"capture_{image_hash}_{method}"

            if st.session_state.get("last_capture_key") != prediction_key:
                with st.spinner("Memproses foto..."):
                    show_prediction(
                        image_pil,
                        method,
                        models,
                        mtcnn,
                    )

                st.session_state["last_capture_key"] = prediction_key

            else:
                st.info(
                    "Foto sudah diproses. Ambil foto baru "
                    "untuk melakukan prediksi lagi."
                )

        except Exception as e:
            st.error(f"Gagal memproses foto: {e}")
            
# =========================================================
# LIVE WEBCAM — FLIP DAN PREDIKSI OTOMATIS
# =========================================================

elif source == "Live Webcam":
    st.info(
        "Izinkan akses kamera. Wajah akan dideteksi dan "
        "diprediksi otomatis saat video berjalan."
    )

    ctx = webrtc_streamer(
        key=f"live-webcam-{method}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: WebcamProcessor(
            models,
            mtcnn,
            method,
        ),
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.state.playing:
        st.caption(
            "Bounding box hijau menandai wajah yang terdeteksi. "
            "Label prediksi diperbarui secara berkala."
        )
    else:
        st.caption("Webcam belum aktif.")