"""
AI Exam Surveillance
Central Configuration

Pipeline:
CCTV
 -> OpenCV
 -> YOLO11m
 -> MediaPipe
 -> GazeNet V2
 -> Face Identity
 -> 37 Feature Extraction
 -> XGBoost
 -> Firebase / ESP32
 -> Administrator Dashboard
"""

import os
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = Path(
    os.getenv("MODELS_DIR", BASE_DIR / "models")
)

DATASETS_DIR = Path(
    os.getenv("DATASETS_DIR", BASE_DIR / "datasets")
)

RESULTS_DIR = Path(
    os.getenv("RESULTS_DIR", BASE_DIR / "results")
)

LOGS_DIR = Path(
    os.getenv("LOGS_DIR", BASE_DIR / "logs")
)


# ============================================================
# AI PIPELINE
# ============================================================

AI_PIPELINE = {
    "object_detection": "YOLO11m",
    "landmarks_head_pose": "MediaPipe",
    "gaze_estimation": "GazeNet V2",
    "identity_verification": "Face Identity Module",
    "feature_extraction": "37-feature representation",
    "behavior_classifier": "XGBoost",
    "behavior_classes": ["Normal", "Cheating"],
    "cloud_logging": "Firebase",
    "iot_alert": "ESP32",
}


# ============================================================
# YOLO11m
# ============================================================

YOLO11M_MODEL = Path(
    os.getenv(
        "YOLO11M_MODEL",
        MODELS_DIR / "yolo11m" / "best.pt"
    )
)

YOLO11M_CONFIDENCE = float(
    os.getenv("YOLO11M_CONFIDENCE", "0.30")
)

YOLO11M_IOU = float(
    os.getenv("YOLO11M_IOU", "0.45")
)

YOLO11M_IMAGE_SIZE = int(
    os.getenv("YOLO11M_IMAGE_SIZE", "640")
)

YOLO11M_MAX_DETECTIONS = int(
    os.getenv("YOLO11M_MAX_DETECTIONS", "50")
)

YOLO11M_DEVICE = os.getenv(
    "YOLO11M_DEVICE",
    "auto"
)


YOLO11M_CLASSES = [
    "phone",
    "book",
    "cheat_paper",
    "calculator",
    "earphone",
    "sunglasses",
    "watch",
    "Answer_paper",
    "laptop",
]


# ============================================================
# RESTRICTED / EXAM OBJECTS
# ============================================================

RESTRICTED_ITEMS = [
    "phone",
    "book",
    "cheat_paper",
    "calculator",
    "earphone",
    "sunglasses",
    "watch",
    "laptop",
]

ALLOWED_ITEMS = [
    "Answer_paper",
]


# ============================================================
# MEDIAPIPE
# ============================================================

MEDIAPIPE_ENABLED = True

MEDIAPIPE_MAX_FACES = int(
    os.getenv("MEDIAPIPE_MAX_FACES", "1")
)

MEDIAPIPE_MAX_HANDS = int(
    os.getenv("MEDIAPIPE_MAX_HANDS", "2")
)

MEDIAPIPE_FACE_CONFIDENCE = float(
    os.getenv("MEDIAPIPE_FACE_CONFIDENCE", "0.50")
)

MEDIAPIPE_HAND_CONFIDENCE = float(
    os.getenv("MEDIAPIPE_HAND_CONFIDENCE", "0.50")
)

MEDIAPIPE_TRACKING_CONFIDENCE = float(
    os.getenv("MEDIAPIPE_TRACKING_CONFIDENCE", "0.50")
)


# ============================================================
# GazeNet V2
# ============================================================

GAZENET_V2_MODEL = Path(
    os.getenv(
        "GAZENET_V2_MODEL",
        MODELS_DIR / "gazenet_v2" / "gazenet_v2_best.pt"
    )
)

GAZENET_V2_LAST_MODEL = Path(
    os.getenv(
        "GAZENET_V2_LAST_MODEL",
        MODELS_DIR / "gazenet_v2" / "gazenet_v2_last.pt"
    )
)

GAZENET_V2_IMAGE_SIZE = int(
    os.getenv("GAZENET_V2_IMAGE_SIZE", "224")
)

GAZENET_V2_DEVICE = os.getenv(
    "GAZENET_V2_DEVICE",
    "auto"
)

GAZENET_V2_DATASET_ID = int(
    os.getenv("GAZENET_V2_DATASET_ID", "0")
)

GAZENET_GAZE_THRESHOLD_DEGREES = float(
    os.getenv("GAZENET_GAZE_THRESHOLD_DEGREES", "15.0")
)


# ============================================================
# FACE IDENTITY
# ============================================================

FACE_IDENTITY_MODEL = Path(
    os.getenv(
        "FACE_IDENTITY_MODEL",
        MODELS_DIR / "face_identity"
    )
)

FACE_REFERENCE_DIR = Path(
    os.getenv(
        "FACE_REFERENCE_DIR",
        DATASETS_DIR / "face_references"
    )
)

FACE_IDENTITY_THRESHOLD = float(
    os.getenv("FACE_IDENTITY_THRESHOLD", "0.89")
)

FACE_EMBEDDING_SIZE = int(
    os.getenv("FACE_EMBEDDING_SIZE", "512")
)


# ============================================================
# XGBOOST
# ============================================================

XGBOOST_MODEL = Path(
    os.getenv(
        "XGBOOST_MODEL",
        MODELS_DIR / "xgboost" / "xgboost_cheating_model.pkl"
    )
)

XGBOOST_PREPROCESSOR = Path(
    os.getenv(
        "XGBOOST_PREPROCESSOR",
        MODELS_DIR / "xgboost" / "xgboost_preprocessor.pkl"
    )
)

XGBOOST_NORMAL_LABEL = 0
XGBOOST_CHEATING_LABEL = 1

BEHAVIOR_CLASSES = [
    "Normal",
    "Cheating",
]

XGBOOST_CHEATING_THRESHOLD = float(
    os.getenv("XGBOOST_CHEATING_THRESHOLD", "0.50")
)


# ============================================================
# EXACT 37 DATASET FEATURES
# ============================================================

FEATURE_NAMES = [
    "id",
    "face_present",
    "no_of_face",
    "face_x",
    "face_y",
    "face_w",
    "face_h",
    "left_eye_x",
    "left_eye_y",
    "right_eye_x",
    "right_eye_y",
    "nose_tip_x",
    "nose_tip_y",
    "mouth_x",
    "mouth_y",
    "face_conf",
    "hand_count",
    "left_hand_x",
    "left_hand_y",
    "right_hand_x",
    "right_hand_y",
    "hand_obj_interaction",
    "head_pose",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "phone_present",
    "phone_loc_x",
    "phone_loc_y",
    "phone_conf",
    "gaze_on_script",
    "gaze_direction",
    "gazePoint_x",
    "gazePoint_y",
    "pupil_left_x",
    "pupil_left_y",
    "pupil_right_x",
    "pupil_right_y",
]

FEATURE_COUNT = len(FEATURE_NAMES)


# ============================================================
# CAMERA
# ============================================================

CAMERA_SOURCE = os.getenv(
    "CAMERA_SOURCE",
    "0"
)

CAMERA_WIDTH = int(
    os.getenv("CAMERA_WIDTH", "1280")
)

CAMERA_HEIGHT = int(
    os.getenv("CAMERA_HEIGHT", "720")
)

CAMERA_FPS = int(
    os.getenv("CAMERA_FPS", "20")
)

CAMERA_RECONNECT_DELAY = float(
    os.getenv("CAMERA_RECONNECT_DELAY", "2.0")
)

CAMERA_BUFFER_SIZE = int(
    os.getenv("CAMERA_BUFFER_SIZE", "1")
)

CAMERA_JPEG_QUALITY = int(
    os.getenv("CAMERA_JPEG_QUALITY", "85")
)


# ============================================================
# EXAM SETTINGS
# ============================================================

EXAM_STARTED = False

EXAM_ID = ""

EXAM_NAME = ""

STUDENT_ID = ""

STUDENT_NAME = ""

EXAM_DURATION_MINUTES = 60

EXAM_START_TIME = None

EXAM_END_TIME = None


# ============================================================
# DETECTION STATE
# ============================================================

DETECTION_STATE = {
    "face_present": False,
    "no_of_face": 0,

    "face_x": 0.0,
    "face_y": 0.0,
    "face_w": 0.0,
    "face_h": 0.0,
    "face_conf": 0.0,

    "left_eye_x": 0.0,
    "left_eye_y": 0.0,
    "right_eye_x": 0.0,
    "right_eye_y": 0.0,

    "nose_tip_x": 0.0,
    "nose_tip_y": 0.0,

    "mouth_x": 0.0,
    "mouth_y": 0.0,

    "hand_count": 0,

    "left_hand_x": 0.0,
    "left_hand_y": 0.0,

    "right_hand_x": 0.0,
    "right_hand_y": 0.0,

    "hand_obj_interaction": 0,

    "head_pose": "center",
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0,

    "phone_present": 0,
    "phone_loc_x": 0.0,
    "phone_loc_y": 0.0,
    "phone_conf": 0.0,

    "gaze_on_script": 0,
    "gaze_direction": "center",
    "gazePoint_x": 0.0,
    "gazePoint_y": 0.0,

    "pupil_left_x": 0.0,
    "pupil_left_y": 0.0,
    "pupil_right_x": 0.0,
    "pupil_right_y": 0.0,
}


# ============================================================
# IDENTITY STATE
# ============================================================

IDENTITY_STATE = {
    "verified": False,
    "confidence": 0.0,
    "student_id": "",
    "student_name": "",
}


# ============================================================
# GAZE STATE
# ============================================================

GAZE_STATE = {
    "direction": "center",
    "gaze_on_script": False,
    "gaze_x": 0.0,
    "gaze_y": 0.0,
    "angular_error": None,
    "confidence": 0.0,
}


# ============================================================
# BEHAVIOR STATE
# ============================================================

BEHAVIOR_STATE = {
    "prediction": "Normal",
    "label": 0,
    "confidence": 0.0,
    "normal_probability": 1.0,
    "cheating_probability": 0.0,
    "features_ready": False,
    "reasons": [],
}


# ============================================================
# VIOLATION STATE
# ============================================================

VIOLATION_STATE = {
    "active": False,
    "count": 0,
    "last_type": "",
    "last_time": None,
    "last_confidence": 0.0,
    "reasons": [],
}

VIOLATION_HISTORY = []

MAX_VIOLATION_HISTORY = int(
    os.getenv("MAX_VIOLATION_HISTORY", "500")
)


# ============================================================
# DETECTED OBJECT STATE
# ============================================================

DETECTED_OBJECTS = []

OBJECT_STATE = {
    class_name: {
        "present": False,
        "confidence": 0.0,
        "x": 0.0,
        "y": 0.0,
        "w": 0.0,
        "h": 0.0,
    }
    for class_name in YOLO11M_CLASSES
}


# ============================================================
# SYSTEM STATE
# ============================================================

SYSTEM_STATE = {
    "camera": {
        "connected": False,
        "source": CAMERA_SOURCE,
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "fps": CAMERA_FPS,
    },

    "ai_pipeline": {
        "yolo11m": False,
        "mediapipe": False,
        "gazenet_v2": False,
        "face_identity": False,
        "xgboost": False,
    },

    "firebase": False,
    "esp32": False,

    "last_frame_time": None,
    "processed_frames": 0,
    "fps": 0.0,
}


# ============================================================
# FIREBASE
# ============================================================

FIREBASE_ENABLED = (
    os.getenv("FIREBASE_ENABLED", "false").lower()
    in ("1", "true", "yes", "on")
)

FIREBASE_CREDENTIALS = os.getenv(
    "FIREBASE_CREDENTIALS",
    ""
)

FIREBASE_DATABASE_URL = os.getenv(
    "FIREBASE_DATABASE_URL",
    ""
)


# ============================================================
# ESP32
# ============================================================

ESP32_ENABLED = (
    os.getenv("ESP32_ENABLED", "false").lower()
    in ("1", "true", "yes", "on")
)

ESP32_URL = os.getenv(
    "ESP32_URL",
    ""
)

ESP32_ALERT_ENDPOINT = os.getenv(
    "ESP32_ALERT_ENDPOINT",
    "/alert"
)

ESP32_TIMEOUT = float(
    os.getenv("ESP32_TIMEOUT", "2.0")
)


# ============================================================
# ALERT SETTINGS
# ============================================================

BUZZER_ENABLED = True

LED_ENABLED = True

ALERT_COOLDOWN_SECONDS = float(
    os.getenv("ALERT_COOLDOWN_SECONDS", "5.0")
)

CHEATING_ALERT_THRESHOLD = float(
    os.getenv("CHEATING_ALERT_THRESHOLD", "0.50")
)


# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO"
)

SAVE_EVIDENCE = (
    os.getenv("SAVE_EVIDENCE", "true").lower()
    in ("1", "true", "yes", "on")
)

EVIDENCE_DIR = Path(
    os.getenv(
        "EVIDENCE_DIR",
        RESULTS_DIR / "evidence"
    )
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_camera_source():
    """
    Return the configured camera source.

    Numeric strings such as "0" are converted to integer
    camera indices. RTSP/HTTP/file paths remain strings.
    """
    source = CAMERA_SOURCE

    if isinstance(source, int):
        return source

    source = str(source).strip()

    if source.isdigit():
        return int(source)

    return source


def get_model_paths():
    """
    Return all AI model/artifact paths.
    """
    return {
        "yolo11m": str(YOLO11M_MODEL),
        "gazenet_v2": str(GAZENET_V2_MODEL),
        "gazenet_v2_last": str(GAZENET_V2_LAST_MODEL),
        "face_identity": str(FACE_IDENTITY_MODEL),
        "xgboost_model": str(XGBOOST_MODEL),
        "xgboost_preprocessor": str(XGBOOST_PREPROCESSOR),
    }


def check_models():
    """
    Check whether required model files/directories exist.
    """
    return {
        "yolo11m": YOLO11M_MODEL.is_file(),
        "gazenet_v2": GAZENET_V2_MODEL.is_file(),
        "face_identity": FACE_IDENTITY_MODEL.exists(),
        "xgboost_model": XGBOOST_MODEL.is_file(),
        "xgboost_preprocessor": XGBOOST_PREPROCESSOR.is_file(),
    }


def get_feature_names():
    """
    Return the exact 37-feature order expected by XGBoost.
    """
    return FEATURE_NAMES.copy()


def get_feature_count():
    """
    Return the number of behavioral features.
    """
    return FEATURE_COUNT


def get_restricted_items():
    """
    Return configured restricted exam objects.
    """
    return RESTRICTED_ITEMS.copy()


def reset_detection_state():
    """
    Reset runtime detection state without changing
    static configuration.
    """
    global DETECTION_STATE
    global IDENTITY_STATE
    global GAZE_STATE
    global BEHAVIOR_STATE
    global VIOLATION_STATE
    global VIOLATION_HISTORY
    global DETECTED_OBJECTS
    global OBJECT_STATE

    DETECTION_STATE = {
        "face_present": False,
        "no_of_face": 0,
        "face_x": 0.0,
        "face_y": 0.0,
        "face_w": 0.0,
        "face_h": 0.0,
        "face_conf": 0.0,
        "left_eye_x": 0.0,
        "left_eye_y": 0.0,
        "right_eye_x": 0.0,
        "right_eye_y": 0.0,
        "nose_tip_x": 0.0,
        "nose_tip_y": 0.0,
        "mouth_x": 0.0,
        "mouth_y": 0.0,
        "hand_count": 0,
        "left_hand_x": 0.0,
        "left_hand_y": 0.0,
        "right_hand_x": 0.0,
        "right_hand_y": 0.0,
        "hand_obj_interaction": 0,
        "head_pose": "center",
        "head_pitch": 0.0,
        "head_yaw": 0.0,
        "head_roll": 0.0,
        "phone_present": 0,
        "phone_loc_x": 0.0,
        "phone_loc_y": 0.0,
        "phone_conf": 0.0,
        "gaze_on_script": 0,
        "gaze_direction": "center",
        "gazePoint_x": 0.0,
        "gazePoint_y": 0.0,
        "pupil_left_x": 0.0,
        "pupil_left_y": 0.0,
        "pupil_right_x": 0.0,
        "pupil_right_y": 0.0,
    }

    IDENTITY_STATE = {
        "verified": False,
        "confidence": 0.0,
        "student_id": "",
        "student_name": "",
    }

    GAZE_STATE = {
        "direction": "center",
        "gaze_on_script": False,
        "gaze_x": 0.0,
        "gaze_y": 0.0,
        "angular_error": None,
        "confidence": 0.0,
    }

    BEHAVIOR_STATE = {
        "prediction": "Normal",
        "label": 0,
        "confidence": 0.0,
        "normal_probability": 1.0,
        "cheating_probability": 0.0,
        "features_ready": False,
        "reasons": [],
    }

    VIOLATION_STATE = {
        "active": False,
        "count": 0,
        "last_type": "",
        "last_time": None,
        "last_confidence": 0.0,
        "reasons": [],
    }

    VIOLATION_HISTORY = []
    DETECTED_OBJECTS = []

    OBJECT_STATE = {
        class_name: {
            "present": False,
            "confidence": 0.0,
            "x": 0.0,
            "y": 0.0,
            "w": 0.0,
            "h": 0.0,
        }
        for class_name in YOLO11M_CLASSES
    }


def reset_exam_state():
    """
    Reset exam session and detection state.
    """
    global EXAM_STARTED
    global EXAM_ID
    global EXAM_NAME
    global STUDENT_ID
    global STUDENT_NAME
    global EXAM_START_TIME
    global EXAM_END_TIME

    EXAM_STARTED = False
    EXAM_ID = ""
    EXAM_NAME = ""
    STUDENT_ID = ""
    STUDENT_NAME = ""
    EXAM_START_TIME = None
    EXAM_END_TIME = None

    reset_detection_state()


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

for directory in (
    MODELS_DIR,
    DATASETS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    EVIDENCE_DIR,
):
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass