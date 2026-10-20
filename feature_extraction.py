import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage.feature import local_binary_pattern


# ============================================================
# 1. KONFIGURASI MEDIAPIPE TASKS FACE LANDMARKER
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)

LANDMARK_COUNT = 478

# PENTING: urutan 22 pasangan ini harus sama persis dengan urutan kolom
# pada CSV yang dipakai untuk melatih model antropometri.
LANDMARK_PAIRS = [
    (172, 397),  # Mandible Width
    (0, 14),     # Upper Vermilion Height
    (14, 17),    # Lower Vermilion Height
    (468, 473),  # Interpupillary Distance (IPD)
    (133, 362),  # Intercanthal Width
    (33, 263),   # Biocular Width
    (98, 327),   # Nasal Width
    (168, 2),    # Nasal Height
    (168, 4),    # Nasal Length
    (61, 291),   # Mouth Width
    (168, 152),  # Morphological Facial Height
    (2, 152),    # Lower Facial Height
    (159, 145),  # PFH Left
    (386, 374),  # PFH Right
    (33, 133),   # HPA Left
    (263, 362),  # HPA Right
    (468, 159),  # Approx MRD1 Left
    (468, 145),  # Approx MRD2 Left
    (473, 386),  # Approx MRD1 Right
    (473, 374),  # Approx MRD2 Right
    (37, 267),   # Approx Philtrum Width
    (2, 0),      # Approx Philtrum Height
]

ANTHROPOMETRY_FEATURE_NAMES = [
    "Mandible Width",
    "Upper Vermilion Height",
    "Lower Vermilion Height",
    "Interpupillary Distance (IPD)",
    "Intercanthal Width",
    "Biocular Width",
    "Nasal Width",
    "Nasal Height",
    "Nasal Length",
    "Mouth Width",
    "Morphological Facial Height",
    "Lower Facial Height",
    "PFH Left",
    "PFH Right",
    "HPA Left",
    "HPA Right",
    "Approx MRD1 Left",
    "Approx MRD2 Left",
    "Approx MRD1 Right",
    "Approx MRD2 Right",
    "Approx Philtrum Width",
    "Approx Philtrum Height",
]

_face_landmarker = None
_LANDMARK_LOCK = threading.Lock()


def _ensure_model_file():
    """Pastikan file face_landmarker.task ada; unduh bila belum ada."""
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
        return

    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = MODEL_PATH.with_suffix(".task.tmp")
        urllib.request.urlretrieve(MODEL_URL, str(tmp_path))
        tmp_path.replace(MODEL_PATH)
    except Exception as exc:
        raise FileNotFoundError(
            f"Model MediaPipe tidak ditemukan di {MODEL_PATH} dan gagal "
            "diunduh otomatis. Letakkan file face_landmarker.task di "
            "folder models."
        ) from exc


def _create_face_landmarker():
    _ensure_model_file()

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _get_face_landmarker():
    """Buat FaceLandmarker saat pertama kali dibutuhkan."""
    global _face_landmarker
    if _face_landmarker is None:
        _face_landmarker = _create_face_landmarker()
    return _face_landmarker


# ============================================================
# 2. DETEKSI LANDMARK
# ============================================================
def detect_landmarks(image_bgr):
    """
    Deteksi landmark MediaPipe Tasks dari satu crop wajah BGR.

    Return:
        List berisi array (478, 3). Jika wajah tidak ditemukan, return [].
        x dan y dalam piksel; z mengikuti skala x MediaPipe.
    """
    if image_bgr is None:
        return []

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr harus berupa numpy.ndarray.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Gambar harus memiliki tiga channel warna.")

    if image_bgr.size == 0:
        return []

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    # Kunci agar halaman utama dan thread webcam tidak memakai landmarker bersamaan
    with _LANDMARK_LOCK:
        result = _get_face_landmarker().detect(mp_image)

    if not result.face_landmarks:
        return []

    all_landmarks = []
    for face in result.face_landmarks:
        points = np.asarray(
            [
                [
                    landmark.x * width,
                    landmark.y * height,
                    landmark.z * width,
                ]
                for landmark in face
            ],
            dtype=np.float32,
        )

        if points.shape == (LANDMARK_COUNT, 3):
            all_landmarks.append(points)

    return all_landmarks


# ============================================================
# 3. ANTROPOMETRI 22 FITUR
# ============================================================
def extract_antropometri22(landmarks):
    """Hitung 22 jarak Euclidean 2D sesuai urutan training."""
    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.shape != (LANDMARK_COUNT, 3):
        raise ValueError(
            f"Landmark harus berukuran ({LANDMARK_COUNT}, 3), "
            f"tetapi diperoleh {landmarks.shape}."
        )

    if not np.isfinite(landmarks).all():
        raise ValueError("Landmark mengandung NaN atau infinity.")

    features = []
    for idx1, idx2 in LANDMARK_PAIRS:
        point1 = landmarks[idx1, :2]
        point2 = landmarks[idx2, :2]
        features.append(float(np.linalg.norm(point1 - point2)))

    features = np.asarray(features, dtype=np.float32)
    if features.shape != (22,):
        raise ValueError(f"Jumlah fitur antropometri salah: {features.shape}")

    return features


# ============================================================
# 4. LBP GRID 3x3
# ============================================================
def extract_lbp(image_bgr):
    """
    Pure LBP: P=8, R=1, method=default, grid 3x3,
    256 bin per cell, normalisasi histogram per cell.
    Return array float32 dengan 2304 fitur.
    """
    if image_bgr is None:
        raise ValueError("Gambar kosong atau tidak valid.")

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr harus berupa numpy.ndarray.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Gambar harus memiliki tiga channel warna.")

    if image_bgr.size == 0:
        raise ValueError("Gambar kosong atau tidak valid.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    grid_rows = 3
    grid_cols = 3
    radius = 1
    n_points = 8 * radius
    num_bins = 256

    if height < grid_rows or width < grid_cols:
        raise ValueError("Ukuran gambar terlalu kecil untuk grid LBP 3x3.")

    lbp_matrix = local_binary_pattern(
        gray, n_points, radius, method="default"
    )

    features = []
    for row in range(grid_rows):
        y_start = row * height // grid_rows
        y_end = (row + 1) * height // grid_rows

        for col in range(grid_cols):
            x_start = col * width // grid_cols
            x_end = (col + 1) * width // grid_cols
            cell = lbp_matrix[y_start:y_end, x_start:x_end]

            hist, _ = np.histogram(
                cell.ravel(),
                bins=num_bins,
                range=(0, 256),
            )
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-6
            features.extend(hist)

    features = np.asarray(features, dtype=np.float32)
    if features.shape != (2304,):
        raise ValueError(f"Jumlah fitur LBP salah: {features.shape}")

    return features


# ============================================================
# 5. TUTUP RESOURCE
# ============================================================
def close_face_landmarker():
    """Tutup resource FaceLandmarker bila aplikasi dihentikan."""
    global _face_landmarker
    with _LANDMARK_LOCK:
        if _face_landmarker is not None:
            _face_landmarker.close()
            _face_landmarker = None