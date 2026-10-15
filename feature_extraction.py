
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

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model Face Landmarker tidak ditemukan: {MODEL_PATH}"
    )


# =====================================================
# MEDIAPIPE TASKS - FACE LANDMARKER 478
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

face_landmarker = vision.FaceLandmarker.create_from_options(
    options
)

# =====================================================
# HELPER ANTROPOMETRI
# =====================================================
def _eye_centers(lm):
    if len(lm) != 478:
        raise ValueError("Diperlukan tepat 478 landmark.")

    return lm[468], lm[473]


def _dist(lm, a, b):
    pa = lm[a] if isinstance(a, (int, np.integer)) else a
    pb = lm[b] if isinstance(b, (int, np.integer)) else b

    return np.linalg.norm(pa - pb)


# =====================================================
# ANTROPOMETRI 22 FITUR
# =====================================================
def extract_antropometri22(landmarks):
    lm = landmarks

    if len(lm) != 478:
        raise ValueError("Antropometri membutuhkan 478 landmark.")

    R, L = _eye_centers(lm)

    return np.array(
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

            # 17-20. MRD (iris -> kelopak)
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


# =====================================================
# LBP
# =====================================================
def _lbp_grid_hist(channel):
    lbp = local_binary_pattern(
        channel, 8, 1, method="default"
    )

    h, w = channel.shape
    features = []

    for r in range(3):
        for c in range(3):
            cell = lbp[
                r * h // 3:(r + 1) * h // 3,
                c * w // 3:(c + 1) * w // 3
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.array(
        _lbp_grid_hist(gray),
        dtype=np.float32
    )  # 2304 fitur


# =====================================================
# CONTOH PENGGUNAAN
# =====================================================
# image = cv2.imread("/kaggle/working/contoh.jpg")
# landmarks = detect_landmarks(image)
#
# if landmarks:
#     lm = landmarks[0]
#     fitur22 = extract_antropometri22(lm)
#     fitur_lbp = extract_lbp(image)
#
#     print("Jumlah landmark:", len(lm))
#     print("Antropometri 22:", fitur22.shape)
#     print("LBP:", fitur_lbp.shape)
# else:
#     print("Wajah tidak terdeteksi.")


# =====================================================
# TUTUP MODEL SETELAH SELESAI MEMPROSES SEMUA GAMBAR
# =====================================================
# face_landmarker.close()
