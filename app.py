import time
import threading
from pathlib import Path

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
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


# =====================================================
# CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FACE_SIZE = (510, 510)

# Nama tampilan -> (file SVM, file scaler, jenis fitur)
MODEL_CONFIG = {
    "LBP": (
        "lbp_own_fair_svm.pkl",
        "lbp_own_fair_scaler.pkl",
        "lbp",
    ),
    "Antropometri": (
        "antro_own_fair_svm.pkl",
        "antro_own_fair_scaler.pkl",
        "antro22",
    ),
}


# =====================================================
# DETEKSI WAJAH UNTUK CROP
# Menggunakan MediaPipe Face Detection untuk bounding box.
# MediaPipe Tasks Face Landmarker dipakai di
# feature_extraction.py untuk fitur antropometri.
# =====================================================
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5,
)


# =====================================================
# LOAD MODEL DAN SCALER
# =====================================================
@st.cache_resource
def load_models():
    loaded = {}

    for name, (svm_file, scaler_file, _) in MODEL_CONFIG.items():
        svm_path = MODEL_DIR / svm_file
        scaler_path = MODEL_DIR / scaler_file

        if not svm_path.is_file():
            raise FileNotFoundError(
                f"File SVM tidak ditemukan: {svm_path}"
            )

        if not scaler_path.is_file():
            raise FileNotFoundError(
                f"File scaler tidak ditemukan: {scaler_path}"
            )

        loaded[name] = {
            "model": joblib.load(svm_path),
            "scaler": joblib.load(scaler_path),
        }

    return loaded


# =====================================================
# DETEKSI DAN CROP SEMUA WAJAH
# =====================================================
def crop_all_faces(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        return []

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = face_detector.process(image_rgb)

    if not result.detections:
        return []

    h, w = image_bgr.shape[:2]
    cropped_faces = []

    for detection in result.detections:
        bbox = detection.location_data.relative_bounding_box

        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            continue

        margin_x = int(bw * 0.30)
        margin_y = int(bh * 0.35)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        size = max(bw + 2 * margin_x, bh + 2 * margin_y)
        half = max(1, size // 2)

        left = max(0, cx - half)
        top = max(0, cy - half)
        right = min(w, cx + half)
        bottom = min(h, cy + half)

        face_crop = image_bgr[top:bottom, left:right]

        if face_crop.size > 0:
            cropped_faces.append(
                (face_crop, (x1, y1, bw, bh))
            )

    return cropped_faces


# =====================================================
# PREPROCESSING TANPA REMOVE BACKGROUND
# =====================================================
def preprocess_face(image_bgr):
    return cv2.resize(
        image_bgr,
        FACE_SIZE,
        interpolation=cv2.INTER_AREA,
    )


# =====================================================
# EKSTRAKSI FITUR
# =====================================================
def extract_features(image_bgr, feature_type):
    if feature_type == "lbp":
        feature = extract_lbp(image_bgr)

    elif feature_type == "antro22":
        # Sesuai signature detect_landmarks(image_bgr)
        # di feature_extraction.py; tidak memakai mode="478".
        landmarks = detect_landmarks(image_bgr)

        if landmarks is None or len(landmarks) == 0:
            return None

        feature = extract_antropometri22(landmarks[0])

    else:
        raise ValueError(
            f"Jenis fitur tidak dikenal: {feature_type}"
        )

    if feature is None:
        return None

    return np.asarray(feature, dtype=np.float64).ravel()


# =====================================================
# PREDIKSI
# =====================================================
def predict(model_data, feature):
    model = model_data["model"]
    scaler = model_data["scaler"]

    feature = np.asarray(
        feature, dtype=np.float64
    ).reshape(1, -1)

    expected = getattr(scaler, "n_features_in_", None)

    if expected is not None and feature.shape[1] != expected:
        return (
            "Dimensi fitur tidak cocok "
            f"({feature.shape[1]} fitur, scaler mengharapkan "
            f"{expected})"
        )

    scaled_feature = scaler.transform(feature)
    return str(model.predict(scaled_feature)[0])


# =====================================================
# LIVE WEBCAM
# =====================================================
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.model_name = "LBP"
        self.model_data = None
        self.results = []
        self.last_run = 0.0
        self.interval = 0.7

    def set_model(self, name, model_data):
        with self.lock:
            if name != self.model_name:
                self.results = []
                self.last_run = 0.0

            self.model_name = name
            self.model_data = model_data

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        with self.lock:
            model_name = self.model_name
            model_data = self.model_data

        now = time.time()

        if (
            model_data is not None
            and now - self.last_run >= self.interval
        ):
            self.last_run = now
            new_results = []

            try:
                feature_type = MODEL_CONFIG[model_name][2]

                for face_crop, bbox in crop_all_faces(image):
                    resized_face = preprocess_face(face_crop)
                    feature = extract_features(
                        resized_face, feature_type
                    )

                    if feature is None:
                        label = "Landmark gagal"
                    else:
                        label = predict(model_data, feature)

                    if label.startswith("Dimensi fitur"):
                        label = "Dimensi fitur salah"

                    new_results.append((*bbox, label))

            except Exception as exc:
                new_results = [
                    (10, 30, 0, 0, f"Error: {exc}")
                ]

            with self.lock:
                self.results = new_results

        with self.lock:
            results = list(self.results)

        for x, y, bw, bh, label in results:
            if bw > 0 and bh > 0:
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                image,
                label[:60],
                (max(0, x), max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )


# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(
    page_title="Klasifikasi Etnis - LBP & Antropometri",
    layout="wide",
)

st.title("Klasifikasi Etnis Berbasis Fitur Wajah")
st.write("Perbandingan model SVM LBP dan Antropometri")

try:
    models = load_models()
except Exception as exc:
    st.error(f"Gagal memuat model/scaler: {exc}")
    st.stop()


# =====================================================
# PILIH METODE INPUT
# =====================================================
option = st.radio(
    "Metode Input",
    ["Upload Foto", "Ambil Foto", "Live Webcam"],
    horizontal=True,
)

image_bgr = None


# =====================================================
# UPLOAD FOTO
# =====================================================
if option == "Upload Foto":
    uploaded_file = st.file_uploader(
        "Upload citra wajah",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_bgr = cv2.cvtColor(
            np.array(pil_image), cv2.COLOR_RGB2BGR
        )


# =====================================================
# AMBIL FOTO
# =====================================================
elif option == "Ambil Foto":
    camera_image = st.camera_input("Ambil foto")

    if camera_image is not None:
        pil_image = Image.open(camera_image).convert("RGB")
        image_bgr = cv2.cvtColor(
            np.array(pil_image), cv2.COLOR_RGB2BGR
        )


# =====================================================
# LIVE WEBCAM
# =====================================================
else:
    live_model = st.selectbox(
        "Pilih model",
        list(MODEL_CONFIG.keys()),
    )

    st.caption("Tekan START dan izinkan akses kamera.")

    ctx = webrtc_streamer(
        key="live-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.set_model(
            live_model, models[live_model]
        )


# =====================================================
# PREDIKSI FOTO
# =====================================================
if image_bgr is not None:
    st.subheader("Citra Asli")

    faces = crop_all_faces(image_bgr)
    vis_image = image_bgr.copy()

    for idx, (_, (x, y, w, h)) in enumerate(faces, 1):
        cv2.rectangle(
            vis_image, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )
        cv2.putText(
            vis_image,
            f"Wajah {idx}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    st.image(
        cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB),
        width=500,
    )

    if not faces:
        st.error("Wajah tidak terdeteksi.")
        st.stop()

    st.success(f"Terdeteksi {len(faces)} wajah.")

    tabs = st.tabs([
        f"Wajah {i + 1}" for i in range(len(faces))
    ])

    for idx, (face_crop, _) in enumerate(faces):
        with tabs[idx]:
            col1, col2 = st.columns(2)
            resized_face = preprocess_face(face_crop)

            with col1:
                st.image(
                    cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
                    caption="Crop wajah",
                    width=250,
                )

            with col2:
                st.image(
                    cv2.cvtColor(
                        resized_face, cv2.COLOR_BGR2RGB
                    ),
                    caption=(
                        "Preprocessing 510 × 510 "
                        "(tanpa remove background)"
                    ),
                    width=250,
                )

            st.subheader("Hasil Prediksi")
            columns = st.columns(2)

            for column, model_name in zip(
                columns, MODEL_CONFIG.keys()
            ):
                with column:
                    feature_type = MODEL_CONFIG[model_name][2]

                    try:
                        with st.spinner(
                            f"Menguji model {model_name}..."
                        ):
                            feature = extract_features(
                                resized_face, feature_type
                            )

                            if feature is None:
                                result = "Landmark gagal"
                            else:
                                result = predict(
                                    models[model_name], feature
                                )

                        st.metric(
                            label=model_name,
                            value=result,
                        )

                    except Exception as exc:
                        st.error(f"{model_name}: {exc}")
