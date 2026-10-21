import hashlib
import threading
from io import BytesIO
from pathlib import Path

import av
import cv2
import joblib
import numpy as np
import streamlit as st
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from feature_extraction import (
    detect_landmarks,
    extract_antropometri22,
    extract_lbp,
)


# =========================================================
# 1. KONFIGURASI
# =========================================================
st.set_page_config(
    page_title="Klasifikasi Etnis Wajah",
    page_icon="👤",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
IMAGE_SIZE = (510, 510)
DETECTION_THRESHOLD = 0.95
WEBCAM_PROCESS_EVERY_N_FRAMES = 10

ANTRO_MODEL_PATH = MODEL_DIR / "antro_own_fair_resize_no_bg_svm.pkl"
ANTRO_SCALER_PATH = MODEL_DIR / "antro_own_fair_resize_no_bg_scaler.pkl"
LBP_MODEL_PATH = MODEL_DIR / "lbp_own_fair_resize_svm.pkl"
LBP_SCALER_PATH = MODEL_DIR / "lbp_own_fair_resize_scaler.pkl"

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}


# =========================================================
# 2. FUNGSI TAMPILAN GAMBAR
# =========================================================
def show_image(image, caption=None, width=640):
    """
    Tampilkan PIL Image atau array RGB menggunakan st.image.
    Untuk array OpenCV BGR, ubah dulu menjadi RGB sebelum memanggil fungsi ini.
    """
    try:
        if isinstance(image, Image.Image):
            image_rgb = np.asarray(image.convert("RGB"))
        else:
            image_rgb = np.asarray(image)

            if image_rgb.ndim == 2:
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
            elif image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
            elif image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
                st.error(f"Format gambar tidak didukung: {image_rgb.shape}")
                return

            if image_rgb.dtype != np.uint8:
                image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        st.image(image_rgb, caption=caption, width=width, channels="RGB")
    except Exception as exc:
        st.error(f"Gambar gagal ditampilkan: {exc}")


# =========================================================
# 3. MEMUAT MODEL DAN MTCNN
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
        if not path.is_file():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"File kosong: {path}")

    return (
        joblib.load(ANTRO_MODEL_PATH),
        joblib.load(ANTRO_SCALER_PATH),
        joblib.load(LBP_MODEL_PATH),
        joblib.load(LBP_SCALER_PATH),
    )


@st.cache_resource
def load_mtcnn():
    return MTCNN(keep_all=True, device="cpu")


# =========================================================
# 4. DETEKSI DAN CROP WAJAH
# =========================================================
def detect_face_boxes(image_pil, mtcnn, threshold=DETECTION_THRESHOLD):
    """Mengembalikan bounding box dan skor deteksi MTCNN, urut kiri ke kanan."""
    image_pil = image_pil.convert("RGB")
    boxes, probabilities = mtcnn.detect(image_pil)

    if boxes is None or probabilities is None:
        return []

    detections = []
    for box, probability in zip(boxes, probabilities):
        if box is None or probability is None:
            continue

        box = np.asarray(box, dtype=np.float64)
        probability = float(probability)

        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        if not np.isfinite(probability) or probability < threshold:
            continue

        x1, y1, x2, y2 = box
        if x2 > x1 and y2 > y1:
            detections.append((box, probability))

    detections.sort(key=lambda item: item[0][0])
    return detections


def crop_face_from_box(image_pil, box):
    """Crop wajah dan resize ke 510x510."""
    image_pil = image_pil.convert("RGB")
    image_width, image_height = image_pil.size

    x1, y1, x2, y2 = map(float, box)
    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        raise ValueError("Bounding box wajah tidak valid.")

    center_x = x1 + box_width / 2
    center_y = y1 + box_height / 2 + box_height * 0.05
    half_side = max(box_width, box_height) * 0.95 / 2

    left = max(0, int(round(center_x - half_side)))
    top = max(0, int(round(center_y - half_side)))
    right = min(image_width, int(round(center_x + half_side)))
    bottom = min(image_height, int(round(center_y + half_side)))

    if right <= left or bottom <= top:
        raise ValueError("Hasil crop wajah kosong atau tidak valid.")

    cropped = image_pil.crop((left, top, right, bottom))
    return cropped.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


# =========================================================
# 5. EKSTRAKSI FITUR DAN PREDIKSI
# =========================================================
def predict_antro(image_bgr, model, scaler):
    landmarks_list = detect_landmarks(image_bgr)

    if not landmarks_list:
        raise ValueError(
            "MediaPipe gagal mendeteksi landmark pada crop wajah."
        )

    features = np.asarray(
        extract_antropometri22(landmarks_list[0]),
        dtype=np.float32,
    ).reshape(1, -1)

    if features.shape[1] != 22:
        raise ValueError(
            f"Model Antropometri memerlukan 22 fitur, ditemukan {features.shape[1]}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Fitur antropometri mengandung NaN atau infinity.")

    return str(model.predict(scaler.transform(features))[0])


def predict_lbp(image_bgr, model, scaler):
    features = np.asarray(
        extract_lbp(image_bgr),
        dtype=np.float32,
    ).reshape(1, -1)

    if features.shape[1] != 2304:
        raise ValueError(
            f"Model LBP memerlukan 2304 fitur, ditemukan {features.shape[1]}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Fitur LBP mengandung NaN atau infinity.")

    return str(model.predict(scaler.transform(features))[0])


def predict_faces(image_pil, method, models, mtcnn):
    """Deteksi dan prediksi setiap wajah secara terpisah."""
    antro_model, antro_scaler, lbp_model, lbp_scaler = models
    detections = detect_face_boxes(image_pil, mtcnn)
    results = []

    for index, (box, detection_probability) in enumerate(detections, start=1):
        result = {
            "index": index,
            "box": box,
            "detection_confidence": detection_probability,
            "crop": None,
            "prediction": None,
            "error": None,
        }

        try:
            cropped = crop_face_from_box(image_pil, box)
            image_bgr = cv2.cvtColor(
                np.asarray(cropped),
                cv2.COLOR_RGB2BGR,
            )

            if method == "Antropometri 22 Fitur":
                prediction = predict_antro(
                    image_bgr, antro_model, antro_scaler
                )
            else:
                prediction = predict_lbp(
                    image_bgr, lbp_model, lbp_scaler
                )

            result["crop"] = cropped
            result["prediction"] = prediction
        except Exception as exc:
            result["error"] = str(exc)

        results.append(result)

    return results


# =========================================================
# 6. VISUALISASI HASIL
# =========================================================
def draw_label(image_bgr, x1, y1, label):
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


def draw_face_boxes(image_pil, results):
    """Mengembalikan gambar RGB dengan kotak dan label prediksi."""
    image_bgr = cv2.cvtColor(
        np.asarray(image_pil.convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )
    height, width = image_bgr.shape[:2]

    for result in results:
        x1, y1, x2, y2 = map(int, result["box"])
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)

        if result["prediction"]:
            label = f"Wajah {result['index']}: {result['prediction']}"
        elif result["error"]:
            label = f"Wajah {result['index']}: gagal"
        else:
            label = f"Wajah {result['index']}"

        draw_label(image_bgr, x1, y1, label)

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def show_face_results(results):
    if not results:
        st.warning(
            "Tidak ada wajah yang terdeteksi dengan confidence MTCNN "
            f"minimal {DETECTION_THRESHOLD:.2f}. Pastikan wajah terlihat jelas. "
            "Jika perlu, ubah DETECTION_THRESHOLD menjadi 0.90."
        )
        return

    st.success(f"Jumlah wajah terdeteksi: {len(results)}")

    for result in results:
        st.markdown(f"### Wajah {result['index']}")
        col_image, col_result = st.columns([1, 2])

        with col_image:
            if result["crop"] is not None:
                show_image(
                    result["crop"],
                    "Crop wajah (510 × 510)",
                    width=300,
                )

        with col_result:
            st.caption(
                "Skor deteksi MTCNN: "
                f"{result['detection_confidence']:.3f}"
            )
            if result["error"]:
                st.error(f"Prediksi gagal: {result['error']}")
            else:
                st.success(f"Hasil prediksi: **{result['prediction']}**")


# =========================================================
# 7. LIVE WEBCAM
# =========================================================
class WebcamProcessor(VideoProcessorBase):
    def __init__(self, models, mtcnn, method):
        self.models = models
        self.mtcnn = mtcnn
        self.method = method
        self.frame_count = 0
        self.last_boxes = []
        self.last_message = "Menunggu deteksi wajah..."
        self.lock = threading.Lock()

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        image_bgr = cv2.flip(image_bgr, 1)
        self.frame_count += 1

        if self.frame_count % WEBCAM_PROCESS_EVERY_N_FRAMES == 1:
            try:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)

                with self.lock:
                    results = predict_faces(
                        image_pil, self.method, self.models, self.mtcnn
                    )

                new_boxes = []
                for result in results:
                    x1, y1, x2, y2 = map(int, result["box"])
                    label = (
                        f"Wajah {result['index']}: {result['prediction']}"
                        if result["prediction"]
                        else f"Wajah {result['index']}: gagal"
                    )
                    new_boxes.append((x1, y1, x2, y2, label))

                self.last_boxes = new_boxes
                self.last_message = "" if results else "Wajah belum terdeteksi"
            except Exception as exc:
                self.last_boxes = []
                self.last_message = f"Error: {str(exc)[:100]}"

        for x1, y1, x2, y2, label in self.last_boxes:
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_label(image_bgr, x1, y1, label)

        if self.last_message:
            cv2.putText(
                image_bgr,
                self.last_message[:100],
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


# =========================================================
# 8. ANTARMUKA STREAMLIT
# =========================================================
st.title("Klasifikasi Etnis Wajah")
st.write(
    "Pilih fitur Antropometri 22 atau LBP 2304. "
    "Setiap wajah yang terdeteksi diproses secara terpisah."
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
            image_pil = ImageOps.exif_transpose(
                Image.open(BytesIO(uploaded_file.getvalue()))
            ).convert("RGB")

            st.subheader("1. Gambar Input")
            show_image(image_pil, "Gambar asli", width=640)

            if st.button("Prediksi", type="primary", key="predict_upload"):
                with st.spinner("Mendeteksi dan memprediksi wajah..."):
                    results = predict_faces(image_pil, method, models, mtcnn)

                st.subheader("2. Hasil Deteksi")
                if results:
                    show_image(
                        draw_face_boxes(image_pil, results),
                        "Deteksi wajah dan hasil prediksi",
                        width=640,
                    )
                show_face_results(results)
        except Exception as exc:
            st.error("Gagal memproses gambar.")
            st.exception(exc)


# ---------------------------------------------------------
# AMBIL FOTO: OTOMATIS DIPREDIKSI
# ---------------------------------------------------------
elif source == "Ambil Foto":
    camera_image = st.camera_input(
        "Ambil foto menggunakan kamera",
        key="capture_image",
    )

    if camera_image is not None:
        try:
            raw_bytes = camera_image.getvalue()
            image_pil = ImageOps.exif_transpose(
                Image.open(BytesIO(raw_bytes))
            ).convert("RGB")

            # Foto ditampilkan dan diproses tanpa dibalik secara horizontal.

            st.subheader("1. Foto Input")
            show_image(image_pil, "Foto yang akan diproses", width=640)

            cache_key = (hashlib.md5(raw_bytes).hexdigest(), method)

            if st.session_state.get("photo_cache_key") != cache_key:
                with st.spinner("Mendeteksi dan memprediksi wajah..."):
                    results = predict_faces(image_pil, method, models, mtcnn)

                st.session_state["photo_cache_key"] = cache_key
                st.session_state["photo_results"] = results
            else:
                results = st.session_state.get("photo_results", [])

            st.subheader("2. Hasil Deteksi")
            if results:
                show_image(
                    draw_face_boxes(image_pil, results),
                    "Deteksi wajah dan hasil prediksi",
                    width=640,
                )
            show_face_results(results)

        except Exception as exc:
            st.error("Gagal memproses foto.")
            st.exception(exc)


# ---------------------------------------------------------
# LIVE WEBCAM
# ---------------------------------------------------------
elif source == "Live Webcam":
    st.info(
        "Klik START dan izinkan akses kamera browser. "
        "Gunakan HTTPS saat deploy agar izin kamera dapat diberikan."
    )

    ctx = webrtc_streamer(
        key=f"live-webcam-{method}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: WebcamProcessor(
            models, mtcnn, method
        ),
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing:
        st.caption(
            "Kotak dan label menunjukkan hasil prediksi terakhir. "
            "Pemrosesan dilakukan berkala untuk mengurangi beban CPU."
        )
    else:
        st.caption(
            "Webcam belum aktif. Periksa izin kamera browser, koneksi HTTPS, "
            "dan pastikan kamera tidak sedang digunakan aplikasi lain."
        )
