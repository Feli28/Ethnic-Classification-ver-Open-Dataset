
import time
import threading
from pathlib import Path

import av
import cv2
import joblib
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
# DETEKSI DAN CROP WAJAH
# MEDIAPIPE TASKS FACE LANDMARKER
# =====================================================
def crop_all_faces(image_bgr):
    """
    Deteksi wajah memakai detect_landmarks() dari
    feature_extraction.py, yang menggunakan FaceLandmarker
    dari file models/face_landmarker.task.

    Return:
        [(face_crop, (x, y, width, height)), ...]
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    landmarks_list = detect_landmarks(image_bgr)

    if landmarks_list is None or len(landmarks_list) == 0:
        return []

    image_h, image_w = image_bgr.shape[:2]
    cropped_faces = []

    for landmarks in landmarks_list:
        points = np.asarray(landmarks, dtype=np.float32)

        if (
            points.ndim != 2
            or points.shape[0] != 478
            or points.shape[1] < 2
        ):
            continue

        # Koordinat landmark dalam piksel
        x_min = int(np.floor(np.min(points[:, 0])))
        y_min = int(np.floor(np.min(points[:, 1])))
        x_max = int(np.ceil(np.max(points[:, 0])))
        y_max = int(np.ceil(np.max(points[:, 1])))

        box_w = x_max - x_min
        box_h = y_max - y_min

        if box_w <= 0 or box_h <= 0:
            continue

        # Margin crop
        margin_x = int(box_w * 0.30)
        margin_y = int(box_h * 0.35)

        left = max(0, x_min - margin_x)
        top = max(0, y_min - margin_y)
        right = min(image_w, x_max + margin_x)
        bottom = min(image_h, y_max + margin_y)

        if right <= left or bottom <= top:
            continue

        face_crop = image_bgr[top:bottom, left:right]

        if face_crop.size == 0:
            continue

        cropped_faces.append(
            (
                face_crop,
                (
                    left,
                    top,
                    right - left,
                    bottom - top,
                ),
            )
        )

    return cropped_faces


# =====================================================
# PREPROCESSING 510 x 510
# TANPA REMOVE BACKGROUND
# =====================================================
def preprocess_face(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Crop wajah kosong.")

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
        landmarks_list = detect_landmarks(image_bgr)

        if not landmarks_list:
            return None

        feature = extract_antropometri22(
            landmarks_list[0]
        )

    else:
        raise ValueError(
            f"Jenis fitur tidak dikenal: {feature_type}"
        )

    if feature is None:
        return None

    return np.asarray(
        feature, dtype=np.float64
    ).ravel()


# =====================================================
# PREDIKSI SVM
# =====================================================
def predict(model_data, feature):
    model = model_data["model"]
    scaler = model_data["scaler"]

    feature = np.asarray(
        feature, dtype=np.float64
    ).reshape(1, -1)

    expected = getattr(
        scaler, "n_features_in_", None
    )

    if (
        expected is not None
        and feature.shape[1] != expected
    ):
        raise ValueError(
            f"Dimensi fitur tidak cocok: diperoleh "
            f"{feature.shape[1]}, sedangkan scaler "
            f"mengharapkan {expected} fitur."
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
                faces = crop_all_faces(image)

                for face_crop, bbox in faces:
                    resized_face = preprocess_face(face_crop)

                    feature = extract_features(
                        resized_face,
                        feature_type,
                    )

                    if feature is None:
                        label = "Landmark gagal"
                    else:
                        label = predict(
                            model_data,
                            feature,
                        )

                    x, y, w, h = bbox
                    new_results.append(
                        (x, y, w, h, label)
                    )

                if not new_results:
                    new_results = [
                        (10, 30, 0, 0, "Wajah tidak terdeteksi")
                    ]

            except Exception as exc:
                new_results = [
                    (10, 30, 0, 0, f"Error: {exc}")
                ]

            with self.lock:
                self.results = new_results

        with self.lock:
            results = list(self.results)

        for x, y, w, h, label in results:
            if w > 0 and h > 0:
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
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
            image,
            format="bgr24",
        )


# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(
    page_title="Klasifikasi Etnis - LBP & Antropometri",
    layout="wide",
)

st.title("Klasifikasi Etnis Berbasis Fitur Wajah")
st.write(
    "Perbandingan model SVM LBP dan Antropometri"
)

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
        try:
            pil_image = Image.open(
                uploaded_file
            ).convert("RGB")

            image_bgr = cv2.cvtColor(
                np.array(pil_image),
                cv2.COLOR_RGB2BGR,
            )
        except Exception as exc:
            st.error(f"Gagal membaca gambar: {exc}")


# =====================================================
# AMBIL FOTO
# =====================================================
elif option == "Ambil Foto":
    camera_image = st.camera_input("Ambil foto")

    if camera_image is not None:
        try:
            pil_image = Image.open(
                camera_image
            ).convert("RGB")

            image_bgr = cv2.cvtColor(
                np.array(pil_image),
                cv2.COLOR_RGB2BGR,
            )
        except Exception as exc:
            st.error(f"Gagal membaca foto: {exc}")


# =====================================================
# LIVE WEBCAM
# =====================================================
else:
    live_model = st.selectbox(
        "Pilih model",
        list(MODEL_CONFIG.keys()),
    )

    st.caption(
        "Tekan START dan izinkan akses kamera."
    )

    ctx = webrtc_streamer(
        key="live-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveProcessor,
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
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
            live_model,
            models[live_model],
        )


# =====================================================
# PREDIKSI FOTO
# =====================================================
if image_bgr is not None:
    st.subheader("Citra Asli")

    try:
        faces = crop_all_faces(image_bgr)
    except Exception as exc:
        st.error(f"Deteksi wajah gagal: {exc}")
        st.stop()

    vis_image = image_bgr.copy()

    for idx, (_, bbox) in enumerate(faces, 1):
        x, y, w, h = bbox

        cv2.rectangle(
            vis_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
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
        cv2.cvtColor(
            vis_image,
            cv2.COLOR_BGR2RGB,
        ),
        caption="Hasil deteksi Face Landmarker",
        width=500,
    )

    if not faces:
        st.warning(
            "Wajah tidak terdeteksi oleh Face Landmarker."
        )
        st.stop()

    st.success(f"Terdeteksi {len(faces)} wajah.")

    tabs = st.tabs(
        [f"Wajah {i + 1}" for i in range(len(faces))]
    )

    for idx, (face_crop, bbox) in enumerate(faces):
        with tabs[idx]:
            resized_face = preprocess_face(face_crop)

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    cv2.cvtColor(
                        face_crop,
                        cv2.COLOR_BGR2RGB,
                    ),
                    caption="Crop wajah",
                    width=250,
                )

            with col2:
                st.image(
                    cv2.cvtColor(
                        resized_face,
                        cv2.COLOR_BGR2RGB,
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
                columns,
                MODEL_CONFIG.keys(),
            ):
                with column:
                    feature_type = MODEL_CONFIG[
                        model_name
                    ][2]

                    try:
                        with st.spinner(
                            f"Menguji model {model_name}..."
                        ):
                            feature = extract_features(
                                resized_face,
                                feature_type,
                            )

                            if feature is None:
                                st.warning(
                                    "Landmark gagal dideteksi."
                                )
                                continue

                            result = predict(
                                models[model_name],
                                feature,
                            )

                        st.metric(
                            label=model_name,
                            value=result,
                        )

                    except Exception as exc:
                        st.error(
                            f"{model_name}: {exc}"
                        )
