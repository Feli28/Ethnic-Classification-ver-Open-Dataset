
import cv2
import numpy as np
import mediapipe as mp

from skimage.feature import local_binary_pattern
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. KONFIGURASI MEDIAPIPE FACE LANDMARKER
# ============================================================

MODEL_PATH = "models/face_landmarker.task"

BaseOptions = python.BaseOptions

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# 2. PASANGAN LANDMARK UNTUK 22 FITUR ANTROPOMETRI
#    Urutan mengikuti kode training
# ============================================================

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
    (2, 0)       # Approx Philtrum Height
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
    "Approx Philtrum Height"
]


# ============================================================
# 3. DETEKSI LANDMARK
# ============================================================

def detect_landmarks(image_bgr):
    """
    Mendeteksi landmark wajah menggunakan MediaPipe Tasks.

    Parameter:
        image_bgr: citra OpenCV dalam format BGR.

    Return:
        List berisi array landmark berbentuk (478, 3).
        Koordinat x dan y dikonversi ke piksel.
        Koordinat z disimpan, tetapi tidak digunakan
        untuk menghitung fitur antropometri.
    """

    if image_bgr is None:
        return []

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError(
            "image_bgr harus berupa numpy.ndarray."
        )

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            "Gambar harus memiliki tiga channel warna."
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    height, width = image_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb)
    )

    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return []

    all_landmarks = []

    for face in result.face_landmarks:
        points = np.array(
            [
                [
                    landmark.x * width,
                    landmark.y * height,
                    landmark.z * width
                ]
                for landmark in face
            ],
            dtype=np.float32
        )

        if points.shape != (478, 3):
            continue

        all_landmarks.append(points)

    return all_landmarks


# ============================================================
# 4. EKSTRAKSI 22 FITUR ANTROPOMETRI
# ============================================================

def extract_antropometri22(landmarks):
    """
    Menghitung 22 fitur antropometri menggunakan jarak
    Euclidean 2D dalam koordinat piksel.

    Urutan pasangan mengikuti kode training.
    Koordinat z tidak digunakan.
    
    Parameter:
        landmarks: array dengan bentuk (478, 3).

    Return:
        Array float32 dengan bentuk (22,).
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    if landmarks.shape != (478, 3):
        raise ValueError(
            "Landmark harus berukuran (478, 3), "
            f"tetapi diperoleh {landmarks.shape}."
        )

    if not np.isfinite(landmarks).all():
        raise ValueError(
            "Landmark mengandung nilai NaN atau tak hingga."
        )

    features = []

    for idx1, idx2 in LANDMARK_PAIRS:
        point1 = landmarks[idx1, :2]
        point2 = landmarks[idx2, :2]

        distance = np.linalg.norm(
            point1 - point2
        )

        features.append(float(distance))

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.shape != (22,):
        raise ValueError(
            f"Jumlah fitur antropometri tidak sesuai: "
            f"{features.shape}."
        )

    return features


# ============================================================
# 5. EKSTRAKSI PURE LBP GRID 3x3
# ============================================================

def extract_lbp(image_bgr):
    """
    Ekstraksi Pure LBP dengan konfigurasi training:
        P = 8
        R = 1
        method = default
        Grid = 3x3
        Histogram = 256 bin per cell
        Normalisasi = hist / (hist.sum() + 1e-6)

    Return:
        Array float32 dengan bentuk (2304,).
    """

    if image_bgr is None:
        raise ValueError(
            "Gambar tidak valid atau kosong."
        )

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError(
            "image_bgr harus berupa numpy.ndarray."
        )

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            "Gambar harus memiliki tiga channel warna."
        )

    image_gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    height, width = image_gray.shape

    GRID_ROWS = 3
    GRID_COLS = 3
    RADIUS = 1
    N_POINTS = 8 * RADIUS
    NUM_BINS = 256

    if height < GRID_ROWS or width < GRID_COLS:
        raise ValueError(
            "Ukuran gambar terlalu kecil untuk grid 3x3."
        )

    lbp_matrix = local_binary_pattern(
        image_gray,
        N_POINTS,
        RADIUS,
        method="default"
    )

    features = []

    for row in range(GRID_ROWS):
        y_start = row * height // GRID_ROWS
        y_end = (row + 1) * height // GRID_ROWS

        for col in range(GRID_COLS):
            x_start = col * width // GRID_COLS
            x_end = (col + 1) * width // GRID_COLS

            cell = lbp_matrix[
                y_start:y_end,
                x_start:x_end
            ]

            hist, _ = np.histogram(
                cell.ravel(),
                bins=NUM_BINS,
                range=(0, 256)
            )

            hist = hist.astype(np.float32)

            hist /= (
                hist.sum() + 1e-6
            )

            features.extend(hist)

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.shape != (2304,):
        raise ValueError(
            f"Jumlah fitur LBP tidak sesuai: "
            f"{features.shape}."
        )

    return features


# ============================================================
# 6. MENUTUP MEDIAPIPE FACE LANDMARKER
# ============================================================

def close_face_landmarker():
    """
    Menutup resource MediaPipe.
    Panggil saat resource tidak lagi digunakan.
    """

    global face_landmarker

    if face_landmarker is not None:
        face_landmarker.close()
        face_landmarker = None
