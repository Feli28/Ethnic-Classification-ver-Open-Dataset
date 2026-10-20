import os
import threading

import av
import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

from feature_extraction import (
    detect_landmarks,
    extract_antropometri22,
    extract_lbp,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Klasifikasi Etnis Wajah",
    page_icon="👤",
    layout="wide",
)

MODEL_DIR = "models"
IMAGE_SIZE = (510, 510)
DETECTION_THRESHOLD = 0.95
WEBCAM_PROCESS_EVERY_N_FRAMES = 10

ANTRO_MODEL_PATH = os.path.join(
    MODEL_DIR, "antro_own_fair_resize_no_bg_svm.pkl"
)
ANTRO_SCALER_PATH = os.path.join(
    MODEL_DIR, "antro_own_fair_resize_no_bg_scaler.pkl"
)
LBP_MODEL_PATH = os.path.join(
    MODEL_DIR, "lbp_own_fair_resize_svm.pkl"
)
LBP_SCALER_PATH = os.path.join(
    MODEL_DIR, "lbp_own_fair_resize_scaler.pkl"
)


# =========================================================
# 2. LOAD MODELS AND MTCNN
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
            raise FileNotFoundError(f"File model tidak ditemukan: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"File model kosong: {path}")

    return (
        joblib.load(ANTRO_MODEL_PATH),
        joblib.load(ANTRO_SCALER_PATH),
        joblib.load(LBP_MODEL_PATH),
        joblib.load(LBP_SCALER_PATH),
    )


@st.cache_resource
def load_mtcnn():
    # keep_all=True agar semua wajah pada satu gambar bisa dideteksi.
    return MTCNN(keep_all=True, device="cpu")


# =========================================================
# 3. DETECT FACE BOXES AND CROP
# =========================================================
def detect_face_boxes(image_pil, mtcnn, threshold=DETECTION_THRESHOLD):
    image_pil = image_pil.convert("RGB")
    boxes, probabilities = mtcnn.detect(image_pil)

    if boxes is None or probabilities is None:
        return []

    detections = []
    for box, probability in zip(boxes, probabilities):
        if box is None or probability is None:
            continue
        if not np.isfinite(box).all() or not np.isfinite(probability):
            continue
        if float(probability) < threshold:
            continue

        x1, y1, x2, y2 = map(float, box)
        if x2 > x1 and y2 > y1:
            detections.append((np.array([x1, y1, x2, y2]), float(probability)))

    # Urutkan dari kiri ke kanan supaya nomor wajah lebih konsisten.
    detections.sort(key=lambda item: item[0][0])
    return detections


def crop_face_from_box(image_pil, box):
    """Crop satu wajah mengikuti pendekatan preprocessing training."""
    image_pil = image_pil.convert("RGB")
    image_width, image_height = image_pil.size

    x1, y1, x2, y2 = map(float, box)
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        raise ValueError("Bounding box wajah tidak valid.")

    center_x = x1 + width / 2
    center_y = y1 + height / 2 + height * 0.05
    half_side = max(width, height) * 0.95 / 2

    left = max(0, int(round(center_x - half_side)))
    top = max(0, int(round(center_y - half_side)))
    right = min(image_width, int(round(center_x + half_side)))
    bottom = min(image_height, int(round(center_y + half_side)))

    if right <= left or bottom <= top:
        raise ValueError("Hasil crop wajah kosong atau tidak valid.")

    cropped = image_pil.crop((left, top, right, bottom))
    return cropped.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


# =========================================================
# 4. FEATURE PREDICTION
# =========================================================
def get_prediction_confidence(model, scaled_features):
    """Confidence hanya ditampilkan jika model mendukung predict_proba."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_features)[0]
        return float(np.max(probabilities))
    return None


def predict_antro(image_bgr, model, scaler):
    landmarks_list = detect_landmarks(image_bgr)
    if not landmarks_list:
        raise ValueError("MediaPipe gagal mendeteksi landmark wajah.")

    landmarks = landmarks_list[0]
    features = np.asarray(
        extract_antropometri22(landmarks), dtype=np.float32
    ).reshape(1, -1)

    if features.shape[1] != 22:
        raise ValueError(
            f"Model antropometri membutuhkan 22 fitur, ditemukan {features.shape[1]}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Fitur antropometri mengandung NaN atau infinity.")

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    confidence = get_prediction_confidence(model, scaled_features)
    return str(prediction), confidence


def predict_lbp(image_bgr, model, scaler):
    features = np.asarray(
        extract_lbp(image_bgr), dtype=np.float32
    ).reshape(1, -1)

    if features.shape[1] != 2304:
        raise ValueError(
            f"Model LBP membutuhkan 2304 fitur, ditemukan {features.shape[1]}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Fitur LBP mengandung NaN atau infinity.")

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    confidence = get_prediction_confidence(model, scaled_features)
    return str(prediction), confidence


# =========================================================
# 5. PROCESS ALL FACES IN A STILL IMAGE
# =========================================================
def predict_faces(image_pil, method, models, mtcnn):
    antro_model, antro_scaler, lbp_model, lbp_scaler = models
    detections = detect_face_boxes(image_pil, mtcnn)
    results = []

    for index, (box, detection_confidence) in enumerate(detections, start=1):
        result = {
            "index": index,
            "box": box,
            "detection_confidence": detection_confidence,
            "crop": None,
            "prediction": None,
            "confidence": None,
            "error": None,
        }

        try:
            cropped = crop_face_from_box(image_pil, box)
            image_bgr = cv2.cvtColor(
                np.asarray(cropped), cv2.COLOR_RGB2BGR
            )

            if method == "Antropometri 22 Fitur":
                prediction, confidence = predict_antro(
                    image_bgr, antro_model, antro_scaler
                )
            else:
                prediction, confidence = predict_lbp(
                    image_bgr, lbp_model, lbp_scaler
                )

            result.update({
                "crop": cropped,
                "prediction": prediction,
                "confidence": confidence,
            })
        except Exception as exc:
            result["error"] = str(exc)

        results.append(result)

    return results


def show_face_results(results):
    if not results:
        st.warning(
            "Tidak ada wajah yang terdeteksi dengan confidence "
            f"minimal {DETECTION_THRESHOLD:.2f}. Coba gunakan foto yang lebih jelas."
        )
        return

    st.success(f"Jumlah wajah terdeteksi: {len(results)}")

    for result in results:
        st.markdown(f"### Wajah {result['index']}")

        col_image, col_result = st.columns([1, 2])
        with col_image:
            if result["crop"] is not None:
                st.image(
                    result["crop"],
                    caption="Crop wajah (510 × 510)",
                    use_container_width=True,
                )

        with col_result:
            st.caption(
                f"Confidence deteksi wajah: "
                f"{result['detection_confidence']:.3f}"
            )

            if result["error"]:
                st.error(f"Prediksi gagal: {result['error']}")
            else:
                st.success(f"Hasil prediksi: **{result['prediction']}**")
                if result["confidence"] is not None:
                    st.caption(
                        "Nilai maksimum predict_proba: "
                        f"{result['confidence']:.2%}. Ini bukan jaminan "
                        "probabilitas prediksi benar."
                    )


def draw_face_boxes(image_pil, results):
    """Gambar semua bounding box dan label prediksi di gambar asli."""
    image = cv2.cvtColor(np.asarray(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    for result in results:
        x1, y1, x2, y2 = map(int, result["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

        if result["prediction"]:
            label = f"Wajah {result['index']}: {result['prediction']}"
        elif result["error"]:
            label = f"Wajah {result['index']}: gagal"
        else:
            label = f"Wajah {result['index']}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, scale, thickness
        )
        label_y = max(y1 - 8, text_height + 8)

        cv2.rectangle(
            image,
            (x1, label_y - text_height - 8),
            (x1 + text_width + 8, label_y + baseline),
            (0, 150, 0),
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

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# =========================================================
# 6. LIVE WEBCAM PROCESSOR (MULTI-FACE)
# =========================================================
class WebcamProcessor(VideoProcessorBase):
    def __init__(self, models, mtcnn, method):
        self.lock = threading.Lock()
        self.models = models
        self.mtcnn = mtcnn
        self.method = method
        self.frame_count = 0
        self.last_boxes = []
        self.last_error = None

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        # Mirror video to behave like a front-facing camera.
        image_bgr = cv2.flip(image_bgr, 1)
        self.frame_count += 1

        if self.frame_count % WEBCAM_PROCESS_EVERY_N_FRAMES == 1:
            try:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                results = predict_faces(
                    image_pil, self.method, self.models, self.mtcnn
                )

                self.last_boxes = []
                for result in results:
                    x1, y1, x2, y2 = map(int, result["box"])
                    if result["prediction"]:
                        label = f"Wajah {result['index']}: {result['prediction']}"
                    elif result["error"]:
                        label = f"Wajah {result['index']}: gagal"
                    else:
                        label = f"Wajah {result['index']}"
                    self.last_boxes.append((x1, y1, x2, y2, label))

                self.last_error = (
                    None if results else "Wajah belum terdeteksi"
                )
            except Exception as exc:
                self.last_error = str(exc)
                self.last_boxes = []

        # Draw the latest detections on each frame.
        for x1, y1, x2, y2, label in self.last_boxes:
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, scale, thickness
            )
            label_y = max(y1 - 8, text_height + 8)
            cv2.rectangle(
                image_bgr,
                (x1, label_y - text_height - 8),
                (x1 + text_width + 8, label_y + baseline),
                (0, 150, 0),
                -1,
            )
            cv2.putText(
                image_bgr,
                label,
                (x1 + 4, label_y - 4),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        if not self.last_boxes and self.last_error:
            cv2.putText(
                image_bgr,
                self.last_error[:70],
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


# =========================================================
# 7. STREAMLIT UI
# =========================================================
st.title("Klasifikasi Etnis Wajah")
st.write(
    "Sistem melakukan prediksi menggunakan fitur antropometri 22 fitur "
    "atau Local Binary Pattern (LBP) 2.304 fitur."
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
except Exception as exc:
    st.error(f"Gagal memuat model atau MTCNN: {exc}")
    st.stop()


# ---------------------------------------------------------
# UPLOAD IMAGE
# ---------------------------------------------------------
if source == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Pilih gambar wajah",
        type=["jpg", "jpeg", "png"],
        key="upload_image",
    )

    if uploaded_file is not None:
        try:
            image_pil = ImageOps.exif_transpose(
                Image.open(uploaded_file)
            ).convert("RGB")

            st.image(
                image_pil,
                caption="Gambar asli",
                use_container_width=True,
            )

            if st.button("Prediksi", type="primary", key="predict_upload"):
                with st.spinner("Mendeteksi dan memprediksi semua wajah..."):
                    results = predict_faces(image_pil, method, models, mtcnn)

                if results:
                    annotated = draw_face_boxes(image_pil, results)
                    st.image(
                        annotated,
                        caption="Deteksi wajah dan hasil prediksi",
                        use_container_width=True,
                    )
                show_face_results(results)

        except Exception as exc:
            st.error(f"Gagal memproses gambar: {exc}")


# ---------------------------------------------------------
# TAKE PHOTO
# ---------------------------------------------------------
elif source == "Ambil Foto":
    camera_image = st.camera_input(
        "Ambil foto menggunakan kamera",
        key="capture_image",
    )

    if camera_image is not None:
        try:
            image_pil = ImageOps.exif_transpose(
                Image.open(camera_image)
            ).convert("RGB")

            # Mirror the captured photo so it matches a mirror-style camera preview.
            image_pil = ImageOps.mirror(image_pil)

            st.image(
                image_pil,
                caption="Foto yang akan diproses",
                use_container_width=True,
            )

            if st.button(
                "Prediksi Semua Wajah",
                type="primary",
                key="predict_capture",
            ):
                with st.spinner("Mendeteksi dan memprediksi semua wajah..."):
                    results = predict_faces(image_pil, method, models, mtcnn)

                if results:
                    annotated = draw_face_boxes(image_pil, results)
                    st.image(
                        annotated,
                        caption="Deteksi wajah dan hasil prediksi",
                        use_container_width=True,
                    )
                show_face_results(results)

        except Exception as exc:
            st.error(f"Gagal memproses foto: {exc}")


# ---------------------------------------------------------
# LIVE WEBCAM
# ---------------------------------------------------------
elif source == "Live Webcam":
    st.info(
        "Izinkan akses kamera. Sistem mencoba mendeteksi dan memprediksi "
        "semua wajah yang terdeteksi. Pemrosesan dilakukan berkala agar "
        "beban CPU tidak terlalu tinggi."
    )

    ctx = webrtc_streamer(
        key=f"live-webcam-{method}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: WebcamProcessor(
            models, mtcnn, method
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing:
        st.caption(
            "Kotak hijau dan label menunjukkan hasil prediksi terakhir. "
            "Karena prediksi dijalankan berkala, label dapat sedikit tertinggal "
            "ketika wajah bergerak."
        )
    else:
        st.caption("Webcam belum aktif.")
