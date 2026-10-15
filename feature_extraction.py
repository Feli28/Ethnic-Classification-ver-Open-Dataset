
import cv2
import numpy as np
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage.feature import local_binary_pattern


# =====================================================
# PATH MODEL FACE LANDMARKER
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"


# =====================================================
# VALIDASI FILE MODEL
# =====================================================
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model Face Landmarker tidak ditemukan: {MODEL_PATH}"
    )


# =====================================================
# MEDIAPIPE TASKS - FACE LANDMARKER
# =====================================================
base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)


# =====================================================
# INISIALISASI FACE LANDMARKER
# =====================================================
try:
    face_landmarker = vision.FaceLandmarker.create_from_options(
        options
    )
except Exception as e:
    raise RuntimeError(
        "Gagal menginisialisasi MediaPipe Face Landmarker. "
        f"{type(e).__name__}: {e}"
    ) from e


# =====================================================
# DETEKSI LANDMARK WAJAH
# =====================================================
def detect_landmarks(image_bgr):
    """
    Input:
        image_bgr: gambar OpenCV dalam format BGR.

    Output:
        List berisi array landmark berukuran (478, 3).
        Mengembalikan list kosong jika wajah tidak terdeteksi.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            "Gambar harus memiliki tiga channel warna (BGR)."
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb)
    )

    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return []

    height, width = image_bgr.shape[:2]
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

        if len(points) == 478:
            all_landmarks.append(points)

    return all_landmarks


# =====================================================
# HELPER ANTROPOMETRI
# =====================================================
def _eye_centers(lm):
    if len(lm) != 478:
        raise ValueError(
            "Diperlukan tepat 478 landmark."
        )

    return lm[468], lm[473]


def _dist(lm, a, b):
    pa = lm[a] if isinstance(a, (int, np.integer)) else a
    pb = lm[b] if isinstance(b, (int, np.integer)) else b

    return np.linalg.norm(pa - pb)


# =====================================================
# EKSTRAKSI ANTROPOMETRI 22 FITUR
# =====================================================
def extract_antropometri22(landmarks):
    lm = np.asarray(landmarks, dtype=np.float32)

    if len(lm) != 478:
        raise ValueError(
            "Antropometri membutuhkan 478 landmark."
        )

    R, L = _eye_centers(lm)

    features = np.array(
        [
            # 1. Mandible Width
            _dist(lm, 172, 397),

            # 2-3. Lip
            _dist(lm, 0, 14),
            _dist(lm, 14, 17),

            # 4-6. Eye
            _dist(lm, R, L),
            _dist(lm, 133, 362),
            _dist(lm, 33, 263),

            # 7-9. Nose
            _dist(lm, 98, 327),
            _dist(lm, 168, 2),
            _dist(lm, 168, 4),

            # 10. Mouth
            _dist(lm, 61, 291),

            # 11-12. Face proportion
            _dist(lm, 168, 152),
            _dist(lm, 2, 152),

            # 13-16. Eye opening
            _dist(lm, 159, 145),
            _dist(lm, 386, 374),
            _dist(lm, 33, 133),
            _dist(lm, 263, 362),

            # 17-20. MRD (iris ke kelopak)
            _dist(lm, R, 159),
            _dist(lm, R, 145),
            _dist(lm, L, 386),
            _dist(lm, L, 374),

            # 21-22. Philtrum
            _dist(lm, 2, 0),
            _dist(lm, 37, 267),
        ],
        dtype=np.float32
    )

    return features


# =====================================================
# EKSTRAKSI FITUR LBP
# =====================================================
def _lbp_grid_hist(channel):
    lbp = local_binary_pattern(
        channel,
        P=8,
        R=1,
        method="default"
    )

    height, width = channel.shape
    features = []

    for row in range(3):
        for col in range(3):
            cell = lbp[
                row * height // 3:
                (row + 1) * height // 3,
                col * width // 3:
                (col + 1) * width // 3
            ]

            hist, _ = np.histogram(
                cell.ravel(),
                bins=256,
                range=(0, 256)
            )

            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-6

            features.extend(hist)

    return features


def extract_lbp(image):
    """
    Input:
        image: gambar OpenCV BGR.

    Output:
        Array LBP dengan 2304 fitur.
    """
    if image is None or image.size == 0:
        raise ValueError(
            "Gambar kosong atau tidak valid."
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    features = np.array(
        _lbp_grid_hist(gray),
        dtype=np.float32
    )

    return features


# =====================================================
# TUTUP MODEL
# =====================================================
def close_face_landmarker():
    face_landmarker.close()
