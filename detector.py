"""
AI Exam Surveillance
Main AI Detection Pipeline

Pipeline:

CCTV Frame
    |
    +--> YOLO11m
    |       |
    |       +--> phone
    |       +--> book
    |       +--> cheat_paper
    |       +--> calculator
    |       +--> earphone
    |       +--> sunglasses
    |       +--> watch
    |       +--> Answer_paper
    |       +--> laptop
    |
    +--> MediaPipe
    |       |
    |       +--> face landmarks
    |       +--> eye landmarks
    |       +--> hand landmarks
    |       +--> head pose
    |
    +--> GazeNet V2
    |
    +--> Face Identity
    |
    +--> Exact 37-feature vector
    |
    +--> XGBoost
    |
    +--> Normal / Cheating
"""

import math
import os
import time
import importlib
from pathlib import Path

import cv2
import numpy as np

import config


# ============================================================
# OPTIONAL IMPORTS
# ============================================================

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as exc:
    YOLO = None
    YOLO_AVAILABLE = False
    print("YOLO import failed:", exc)


try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception as exc:
    mp = None
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe import failed:", exc)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms

    TORCH_AVAILABLE = True

except Exception as exc:
    torch = None
    nn = None
    F = None
    models = None
    transforms = None
    TORCH_AVAILABLE = False
    print("PyTorch import failed:", exc)


try:
    from xgboost_engine import XGBoostEngine

    XGBOOST_AVAILABLE = True

except Exception as exc:
    XGBoostEngine = None
    XGBOOST_AVAILABLE = False
    print("XGBoost engine import failed:", exc)


# ============================================================
# FACE IDENTITY
# ============================================================

FACE_IDENTITY_AVAILABLE = False

_face_identity_module = None

for module_name in (
    "face_identity",
    "face_identity_engine",
    "face_verification",
):

    try:

        _face_identity_module = importlib.import_module(
            module_name
        )

        required_functions = (
            "load_face_model",
            "create_embedding_model",
            "load_reference_embedding",
            "verify_face",
        )

        if all(
            hasattr(
                _face_identity_module,
                name
            )
            for name in required_functions
        ):

            FACE_IDENTITY_AVAILABLE = True
            break

    except Exception:
        continue


if FACE_IDENTITY_AVAILABLE:

    load_face_model = (
        _face_identity_module.load_face_model
    )

    create_embedding_model = (
        _face_identity_module.create_embedding_model
    )

    load_reference_embedding = (
        _face_identity_module.load_reference_embedding
    )

    verify_face = (
        _face_identity_module.verify_face
    )


# ============================================================
# DEVICE
# ============================================================

if TORCH_AVAILABLE:

    if (
        config.YOLO11M_DEVICE == "cpu"
        or config.GAZENET_V2_DEVICE == "cpu"
    ):

        DEVICE = torch.device("cpu")

    elif torch.cuda.is_available():

        DEVICE = torch.device("cuda")

    else:

        DEVICE = torch.device("cpu")

else:

    DEVICE = None


# ============================================================
# YOLO11m
# ============================================================

yolo_model = None


def load_yolo_model():

    global yolo_model

    if not YOLO_AVAILABLE:

        return False

    model_path = Path(
        config.YOLO11M_MODEL
    )

    if not model_path.is_file():

        print(
            "YOLO11m model not found:",
            model_path
        )

        return False

    try:

        print(
            "Loading YOLO11m:",
            model_path
        )

        yolo_model = YOLO(
            str(model_path)
        )

        print(
            "YOLO11m loaded."
        )

        return True

    except Exception as exc:

        print(
            "YOLO11m loading failed:",
            exc
        )

        yolo_model = None

        return False


# ============================================================
# GAZENET V2
# ============================================================

gaze_model = None
gaze_transform = None


if TORCH_AVAILABLE:

    class GazeNetV2(nn.Module):
        """
        Exact inference architecture corresponding to the
        GazeNet V2 training source.
        """

        def __init__(self):

            super().__init__()

            backbone = models.resnet18(
                weights=None
            )

            feature_dim = (
                backbone.fc.in_features
            )

            backbone.fc = nn.Identity()

            self.backbone = backbone

            self.dataset_embedding = (
                nn.Embedding(
                    2,
                    16
                )
            )

            fusion_dim = (
                feature_dim + 16
            )

            self.shared = nn.Sequential(
                nn.Linear(
                    fusion_dim,
                    256
                ),

                nn.ReLU(),

                nn.BatchNorm1d(
                    256
                ),

                nn.Dropout(
                    0.35
                ),
            )

            self.gaze_head_mpiigaze = (
                nn.Sequential(
                    nn.Linear(
                        256,
                        128
                    ),

                    nn.ReLU(),

                    nn.Dropout(
                        0.25
                    ),

                    nn.Linear(
                        128,
                        3
                    )
                )
            )

            self.gaze_head_facegaze = (
                nn.Sequential(
                    nn.Linear(
                        256,
                        128
                    ),

                    nn.ReLU(),

                    nn.Dropout(
                        0.25
                    ),

                    nn.Linear(
                        128,
                        3
                    )
                )
            )

            self.screen_head = (
                nn.Sequential(
                    nn.Linear(
                        256,
                        64
                    ),

                    nn.ReLU(),

                    nn.Dropout(
                        0.20
                    ),

                    nn.Linear(
                        64,
                        2
                    )
                )
            )


        def forward(
            self,
            x,
            dataset_id
        ):

            features = self.backbone(
                x
            )

            domain = (
                self.dataset_embedding(
                    dataset_id
                )
            )

            features = torch.cat(
                [
                    features,
                    domain
                ],
                dim=1
            )

            features = self.shared(
                features
            )

            gaze_mpii = (
                self.gaze_head_mpiigaze(
                    features
                )
            )

            gaze_face = (
                self.gaze_head_facegaze(
                    features
                )
            )

            dataset_mask = (
                dataset_id == 0
            ).unsqueeze(1)

            gaze = torch.where(
                dataset_mask,
                gaze_mpii,
                gaze_face
            )

            gaze = F.normalize(
                gaze,
                p=2,
                dim=1
            )

            screen = self.screen_head(
                features
            )

            screen = torch.tanh(
                screen
            )

            return gaze, screen


def load_gazenet():

    global gaze_model
    global gaze_transform

    if not TORCH_AVAILABLE:

        return False

    model_path = Path(
        config.GAZENET_V2_MODEL
    )

    if not model_path.is_file():

        print(
            "GazeNet V2 checkpoint not found:",
            model_path
        )

        return False

    try:

        print(
            "Loading GazeNet V2:",
            model_path
        )

        gaze_model = GazeNetV2()

        checkpoint = torch.load(
            str(model_path),
            map_location=DEVICE
        )

        if isinstance(
            checkpoint,
            dict
        ):

            state_dict = (
                checkpoint.get(
                    "model_state_dict"
                )
                or
                checkpoint.get(
                    "state_dict"
                )
                or
                checkpoint
            )

        else:

            state_dict = checkpoint

        cleaned_state_dict = {}

        for key, value in (
            state_dict.items()
        ):

            new_key = key

            if new_key.startswith(
                "module."
            ):

                new_key = new_key[
                    len("module.") :
                ]

            cleaned_state_dict[
                new_key
            ] = value

        gaze_model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

        gaze_model.to(
            DEVICE
        )

        gaze_model.eval()

        imagenet_mean = [
            0.485,
            0.456,
            0.406
        ]

        imagenet_std = [
            0.229,
            0.224,
            0.225
        ]

        gaze_transform = (
            transforms.Compose(
                [
                    transforms.ToPILImage(),

                    transforms.Resize(
                        (
                            config.GAZENET_V2_IMAGE_SIZE,
                            config.GAZENET_V2_IMAGE_SIZE,
                        )
                    ),

                    transforms.ToTensor(),

                    transforms.Normalize(
                        imagenet_mean,
                        imagenet_std
                    ),
                ]
            )
        )

        print(
            "GazeNet V2 loaded."
        )

        return True

    except Exception as exc:

        print(
            "GazeNet V2 loading failed:",
            exc
        )

        gaze_model = None

        return False


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = None
mp_hands = None

face_mesh = None
hands_detector = None


def initialize_mediapipe():

    global mp_face_mesh
    global mp_hands
    global face_mesh
    global hands_detector

    if not MEDIAPIPE_AVAILABLE:

        return False

    try:

        mp_face_mesh = (
            mp.solutions.face_mesh
        )

        mp_hands = (
            mp.solutions.hands
        )

        face_mesh = (
            mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=(
                    config.MEDIAPIPE_MAX_FACES
                ),
                refine_landmarks=True,
                min_detection_confidence=(
                    config.MEDIAPIPE_FACE_CONFIDENCE
                ),
                min_tracking_confidence=(
                    config.MEDIAPIPE_TRACKING_CONFIDENCE
                ),
            )
        )

        hands_detector = (
            mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=(
                    config.MEDIAPIPE_MAX_HANDS
                ),
                min_detection_confidence=(
                    config.MEDIAPIPE_HAND_CONFIDENCE
                ),
                min_tracking_confidence=(
                    config.MEDIAPIPE_TRACKING_CONFIDENCE
                ),
            )
        )

        print(
            "MediaPipe initialized."
        )

        return True

    except Exception as exc:

        print(
            "MediaPipe initialization failed:",
            exc
        )

        face_mesh = None
        hands_detector = None

        return False


# ============================================================
# FACE IDENTITY
# ============================================================

face_model = None
embedding_model = None
reference_embedding = None


def initialize_face_identity():

    global face_model
    global embedding_model
    global reference_embedding

    if not FACE_IDENTITY_AVAILABLE:

        config.SYSTEM_STATE[
            "ai_pipeline"
        ]["face_identity"] = False

        return False

    try:

        face_model = (
            load_face_model()
        )

        embedding_model = (
            create_embedding_model(
                face_model
            )
        )

        reference_embedding = (
            load_reference_embedding(
                embedding_model
            )
        )

        config.SYSTEM_STATE[
            "ai_pipeline"
        ]["face_identity"] = True

        print(
            "Face Identity initialized."
        )

        return True

    except Exception as exc:

        print(
            "Face Identity initialization failed:",
            exc
        )

        face_model = None
        embedding_model = None
        reference_embedding = None

        return False


# ============================================================
# XGBOOST
# ============================================================

xgboost_engine = None


def initialize_xgboost():

    global xgboost_engine

    if not XGBOOST_AVAILABLE:

        return False

    try:

        xgboost_engine = (
            XGBoostEngine()
        )

        ready = bool(
            xgboost_engine.loaded
        )

        config.SYSTEM_STATE[
            "ai_pipeline"
        ]["xgboost"] = ready

        print(
            "XGBoost:",
            "READY" if ready else "NOT READY"
        )

        return ready

    except Exception as exc:

        print(
            "XGBoost initialization failed:",
            exc
        )

        xgboost_engine = None

        return False


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def clamp(
    value,
    minimum,
    maximum
):

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


def normalize_label(
    label
):

    return (
        str(label)
        .strip()
        .lower()
        .replace(
            " ",
            "_"
        )
    )


def box_area(
    box
):

    x1, y1, x2, y2 = box

    return max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )


def box_center(
    box
):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    intersection = max(
        0,
        ix2 - ix1
    ) * max(
        0,
        iy2 - iy1
    )

    union = (
        box_area(box_a)
        +
        box_area(box_b)
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection / union
    )


# ============================================================
# YOLO DETECTION
# ============================================================

def run_yolo(
    frame
):

    if yolo_model is None:

        return []

    try:

        results = yolo_model.predict(
            source=frame,
            conf=config.YOLO11M_CONFIDENCE,
            iou=config.YOLO11M_IOU,
            imgsz=config.YOLO11M_IMAGE_SIZE,
            max_det=config.YOLO11M_MAX_DETECTIONS,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        names = result.names

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:

            try:

                coordinates = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

                confidence = float(
                    box.conf[0]
                    .detach()
                    .cpu()
                    .item()
                )

                class_id = int(
                    box.cls[0]
                    .detach()
                    .cpu()
                    .item()
                )

                if isinstance(
                    names,
                    dict
                ):

                    label = names.get(
                        class_id,
                        str(class_id)
                    )

                else:

                    label = names[
                        class_id
                    ]

                x1, y1, x2, y2 = (
                    map(
                        float,
                        coordinates
                    )
                )

                detections.append(
                    {
                        "label": str(
                            label
                        ),
                        "class_id": class_id,
                        "confidence": confidence,
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "center": box_center(
                            (
                                x1,
                                y1,
                                x2,
                                y2
                            )
                        ),
                        "area_ratio": 0.0,
                    }
                )

            except Exception:
                continue

        height, width = (
            frame.shape[:2]
        )

        frame_area = (
            width * height
        )

        if frame_area > 0:

            for detection in detections:

                detection[
                    "area_ratio"
                ] = (
                    box_area(
                        detection["box"]
                    )
                    /
                    frame_area
                )

        return detections

    except Exception as exc:

        print(
            "YOLO inference error:",
            exc
        )

        return []


# ============================================================
# MEDIAPIPE LANDMARKS
# ============================================================

def run_mediapipe(
    frame
):

    output = {
        "face_landmarks": [],
        "hand_landmarks": [],
    }

    if (
        face_mesh is None
        and
        hands_detector is None
    ):

        return output

    try:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        if face_mesh is not None:

            face_result = (
                face_mesh.process(
                    rgb
                )
            )

            if (
                face_result.multi_face_landmarks
            ):

                output[
                    "face_landmarks"
                ] = list(
                    face_result.multi_face_landmarks
                )

        if hands_detector is not None:

            hand_result = (
                hands_detector.process(
                    rgb
                )
            )

            if (
                hand_result.multi_hand_landmarks
            ):

                output[
                    "hand_landmarks"
                ] = list(
                    hand_result.multi_hand_landmarks
                )

    except Exception as exc:

        print(
            "MediaPipe inference error:",
            exc
        )

    return output


# ============================================================
# LANDMARK POINT
# ============================================================

def landmark_xy(
    landmarks,
    index,
    width,
    height
):

    try:

        point = landmarks[
            index
        ]

        return (
            float(
                point.x * width
            ),
            float(
                point.y * height
            )
        )

    except Exception:

        return (
            0.0,
            0.0
        )


# ============================================================
# FACE LANDMARK EXTRACTION
# ============================================================

# MediaPipe Face Mesh landmark indices
NOSE_TIP = 1

MOUTH_LEFT = 61
MOUTH_RIGHT = 291

LEFT_EYE_CENTER = 468
RIGHT_EYE_CENTER = 473

LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263


def extract_face_data(
    face_landmarks,
    width,
    height
):

    data = {
        "face_present": 0,
        "no_of_face": 0,

        "face_x": 0.0,
        "face_y": 0.0,
        "face_w": 0.0,
        "face_h": 0.0,

        "left_eye_x": 0.0,
        "left_eye_y": 0.0,

        "right_eye_x": 0.0,
        "right_eye_y": 0.0,

        "nose_tip_x": 0.0,
        "nose_tip_y": 0.0,

        "mouth_x": 0.0,
        "mouth_y": 0.0,

        "face_conf": 0.0,
    }

    if not face_landmarks:

        return data

    face = face_landmarks[0]

    points = face.landmark

    xs = [
        float(
            point.x * width
        )
        for point in points
    ]

    ys = [
        float(
            point.y * height
        )
        for point in points
    ]

    if not xs or not ys:

        return data

    x1 = clamp(
        min(xs),
        0,
        width - 1
    )

    y1 = clamp(
        min(ys),
        0,
        height - 1
    )

    x2 = clamp(
        max(xs),
        0,
        width - 1
    )

    y2 = clamp(
        max(ys),
        0,
        height - 1
    )

    left_eye = landmark_xy(
        points,
        LEFT_EYE_CENTER,
        width,
        height
    )

    right_eye = landmark_xy(
        points,
        RIGHT_EYE_CENTER,
        width,
        height
    )

    nose = landmark_xy(
        points,
        NOSE_TIP,
        width,
        height
    )

    mouth_left = landmark_xy(
        points,
        MOUTH_LEFT,
        width,
        height
    )

    mouth_right = landmark_xy(
        points,
        MOUTH_RIGHT,
        width,
        height
    )

    mouth = (
        (
            mouth_left[0]
            +
            mouth_right[0]
        ) / 2.0,

        (
            mouth_left[1]
            +
            mouth_right[1]
        ) / 2.0
    )

    data.update(
        {
            "face_present": 1,
            "no_of_face": 1,

            "face_x": x1,
            "face_y": y1,
            "face_w": max(
                0.0,
                x2 - x1
            ),
            "face_h": max(
                0.0,
                y2 - y1
            ),

            "left_eye_x": left_eye[0],
            "left_eye_y": left_eye[1],

            "right_eye_x": right_eye[0],
            "right_eye_y": right_eye[1],

            "nose_tip_x": nose[0],
            "nose_tip_y": nose[1],

            "mouth_x": mouth[0],
            "mouth_y": mouth[1],

            "face_conf": 1.0,
        }
    )

    return data


# ============================================================
# HAND LANDMARK EXTRACTION
# ============================================================

def extract_hand_data(
    hand_landmarks,
    width,
    height
):

    data = {
        "hand_count": 0,

        "left_hand_x": 0.0,
        "left_hand_y": 0.0,

        "right_hand_x": 0.0,
        "right_hand_y": 0.0,
    }

    if not hand_landmarks:

        return data

    data[
        "hand_count"
    ] = min(
        len(hand_landmarks),
        2
    )

    centers = []

    for hand in hand_landmarks:

        try:

            wrist = hand.landmark[
                0
            ]

            centers.append(
                (
                    float(
                        wrist.x * width
                    ),
                    float(
                        wrist.y * height
                    )
                )
            )

        except Exception:
            continue

    if len(centers) >= 1:

        data[
            "left_hand_x"
        ] = centers[0][0]

        data[
            "left_hand_y"
        ] = centers[0][1]

    if len(centers) >= 2:

        data[
            "right_hand_x"
        ] = centers[1][0]

        data[
            "right_hand_y"
        ] = centers[1][1]

    return data


# ============================================================
# HEAD POSE
# ============================================================

def estimate_head_pose(
    face_landmarks,
    width,
    height
):

    if not face_landmarks:

        return {
            "head_pose": "center",
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
        }

    points = face_landmarks[
        0
    ].landmark

    try:

        left_eye = landmark_xy(
            points,
            LEFT_EYE_CENTER,
            width,
            height
        )

        right_eye = landmark_xy(
            points,
            RIGHT_EYE_CENTER,
            width,
            height
        )

        nose = landmark_xy(
            points,
            NOSE_TIP,
            width,
            height
        )

        eye_center = (
            (
                left_eye[0]
                +
                right_eye[0]
            ) / 2.0,

            (
                left_eye[1]
                +
                right_eye[1]
            ) / 2.0
        )

        eye_distance = math.hypot(
            right_eye[0]
            -
            left_eye[0],

            right_eye[1]
            -
            left_eye[1]
        )

        if eye_distance <= 1e-6:

            return {
                "head_pose": "center",
                "head_pitch": 0.0,
                "head_yaw": 0.0,
                "head_roll": 0.0,
            }

        yaw = (
            (
                nose[0]
                -
                eye_center[0]
            )
            /
            eye_distance
        ) * 60.0

        pitch = (
            (
                nose[1]
                -
                eye_center[1]
            )
            /
            eye_distance
        )

        pitch = (
            pitch - 0.35
        ) * 80.0

        roll = math.degrees(
            math.atan2(
                right_eye[1]
                -
                left_eye[1],

                right_eye[0]
                -
                left_eye[0]
            )
        )

        if abs(yaw) > 20:

            direction = (
                "right"
                if yaw > 0
                else "left"
            )

        elif abs(pitch) > 20:

            direction = (
                "down"
                if pitch > 0
                else "up"
            )

        else:

            direction = "center"

        return {
            "head_pose": direction,
            "head_pitch": float(
                pitch
            ),
            "head_yaw": float(
                yaw
            ),
            "head_roll": float(
                roll
            ),
        }

    except Exception:

        return {
            "head_pose": "center",
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
        }


# ============================================================
# GAZE ESTIMATION
# ============================================================

def run_gazenet(
    face_crop
):

    if (
        gaze_model is None
        or
        gaze_transform is None
        or
        face_crop is None
        or
        face_crop.size == 0
    ):

        return {
            "gaze": (
                0.0,
                0.0,
                1.0
            ),
            "screen": (
                0.0,
                0.0
            ),
            "angular_error": None,
        }

    try:

        rgb = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2RGB
        )

        tensor = gaze_transform(
            rgb
        ).unsqueeze(0)

        tensor = tensor.to(
            DEVICE
        )

        dataset_id = torch.tensor(
            [0],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.no_grad():

            gaze, screen = (
                gaze_model(
                    tensor,
                    dataset_id
                )
            )

        gaze = (
            gaze[0]
            .detach()
            .cpu()
            .numpy()
        )

        screen = (
            screen[0]
            .detach()
            .cpu()
            .numpy()
        )

        norm = np.linalg.norm(
            gaze
        )

        if norm > 1e-8:

            gaze = (
                gaze / norm
            )

        return {
            "gaze": (
                float(gaze[0]),
                float(gaze[1]),
                float(gaze[2])
            ),

            "screen": (
                float(screen[0]),
                float(screen[1])
            ),

            "angular_error": None,
        }

    except Exception as exc:

        print(
            "GazeNet inference error:",
            exc
        )

        return {
            "gaze": (
                0.0,
                0.0,
                1.0
            ),
            "screen": (
                0.0,
                0.0
            ),
            "angular_error": None,
        }


# ============================================================
# GAZE DIRECTION
# ============================================================

def gaze_direction_from_vector(
    gaze
):

    x, y, z = gaze

    horizontal = math.degrees(
        math.atan2(
            x,
            max(
                abs(z),
                1e-6
            )
        )
    )

    vertical = math.degrees(
        math.atan2(
            y,
            max(
                abs(z),
                1e-6
            )
        )
    )

    if abs(horizontal) > 20:

        return (
            "right"
            if horizontal > 0
            else "left"
        )

    if abs(vertical) > 20:

        return (
            "down"
            if vertical > 0
            else "up"
        )

    return "center"


# ============================================================
# PUPIL FEATURES
# ============================================================

def pupil_features(
    face_landmarks,
    width,
    height
):

    result = {
        "pupil_left_x": 0.0,
        "pupil_left_y": 0.0,
        "pupil_right_x": 0.0,
        "pupil_right_y": 0.0,
    }

    if not face_landmarks:
        return result

    points = face_landmarks[
        0
    ].landmark

    left = landmark_xy(
        points,
        LEFT_EYE_CENTER,
        width,
        height
    )

    right = landmark_xy(
        points,
        RIGHT_EYE_CENTER,
        width,
        height
    )

    result.update(
        {
            "pupil_left_x": left[0],
            "pupil_left_y": left[1],

            "pupil_right_x": right[0],
            "pupil_right_y": right[1],
        }
    )

    return result


# ============================================================
# FACE CROP
# ============================================================

def crop_face_from_landmarks(
    frame,
    face_landmarks
):

    if not face_landmarks:

        return None

    height, width = (
        frame.shape[:2]
    )

    points = (
        face_landmarks[0].landmark
    )

    xs = [
        float(
            p.x * width
        )
        for p in points
    ]

    ys = [
        float(
            p.y * height
        )
        for p in points
    ]

    if not xs or not ys:

        return None

    x1 = int(
        clamp(
            min(xs),
            0,
            width - 1
        )
    )

    y1 = int(
        clamp(
            min(ys),
            0,
            height - 1
        )
    )

    x2 = int(
        clamp(
            max(xs),
            x1 + 1,
            width
        )
    )

    y2 = int(
        clamp(
            max(ys),
            y1 + 1,
            height
        )
    )

    return frame[
        y1:y2,
        x1:x2
    ]


# ============================================================
# FACE IDENTITY
# ============================================================

def run_face_identity(
    face_crop
):

    if (
        embedding_model is None
        or
        reference_embedding is None
        or
        face_crop is None
        or
        face_crop.size == 0
    ):

        return {
            "verified": False,
            "similarity": 0.0,
            "identity": "UNKNOWN",
        }

    try:

        result = verify_face(
            embedding_model,
            reference_embedding,
            face_crop,
            threshold=(
                config.FACE_IDENTITY_THRESHOLD
            )
        )

        if result is None:
            result = {}

        return {
            "verified": bool(
                result.get(
                    "verified",
                    False
                )
            ),

            "similarity": float(
                result.get(
                    "similarity",
                    0.0
                )
            ),

            "identity": result.get(
                "identity",
                "UNKNOWN"
            ),
        }

    except Exception as exc:

        print(
            "Face verification error:",
            exc
        )

        return {
            "verified": False,
            "similarity": 0.0,
            "identity": "UNKNOWN",
        }


# ============================================================
# OBJECT FEATURES
# ============================================================

def object_feature_data(
    detections
):

    result = {
        "phone_present": 0,
        "phone_loc_x": 0.0,
        "phone_loc_y": 0.0,
        "phone_conf": 0.0,
        "hand_obj_interaction": 0,
    }

    object_detections = []

    for detection in detections:

        label = normalize_label(
            detection.get(
                "label",
                ""
            )
        )

        if label in (
            normalize_label(
                name
            )
            for name in config.YOLO11M_CLASSES
        ):

            object_detections.append(
                detection
            )

        if label == "phone":

            result[
                "phone_present"
            ] = 1

            center = detection[
                "center"
            ]

            result[
                "phone_loc_x"
            ] = float(
                center[0]
            )

            result[
                "phone_loc_y"
            ] = float(
                center[1]
            )

            result[
                "phone_conf"
            ] = float(
                detection.get(
                    "confidence",
                    0.0
                )
            )

    # --------------------------------------------------------
    # Hand/object interaction
    # --------------------------------------------------------

    hands = [
        d
        for d in detections
        if d.get(
            "type"
        ) == "hand"
    ]

    restricted_objects = [
        d
        for d in object_detections
        if normalize_label(
            d.get(
                "label",
                ""
            )
        )
        in {
            normalize_label(
                item
            )
            for item
            in config.RESTRICTED_ITEMS
        }
    ]

    for hand in hands:

        hand_box = hand.get(
            "box"
        )

        if hand_box is None:
            continue

        for obj in restricted_objects:

            if iou(
                hand_box,
                obj["box"]
            ) > 0.01:

                result[
                    "hand_obj_interaction"
                ] = 1

                break

        if result[
            "hand_obj_interaction"
        ]:

            break

    return result


# ============================================================
# EXACT 37 FEATURES
# ============================================================

def build_xgboost_features(
    detections,
    frame,
    mediapipe_data=None
):

    feature_names = (
        config.FEATURE_NAMES
    )

    features = {
        name: 0
        for name in feature_names
    }

    # Dataset row identifier.
    features["id"] = int(
        time.time() * 1000
    ) % 2147483647

    height, width = (
        frame.shape[:2]
    )

    if mediapipe_data is None:

        mediapipe_data = {
            "face_landmarks": [],
            "hand_landmarks": [],
        }

    face_landmarks = (
        mediapipe_data[
            "face_landmarks"
        ]
    )

    hand_landmarks = (
        mediapipe_data[
            "hand_landmarks"
        ]
    )

    # --------------------------------------------------------
    # Face
    # --------------------------------------------------------

    face_data = extract_face_data(
        face_landmarks,
        width,
        height
    )

    features.update(
        face_data
    )

    # --------------------------------------------------------
    # Hands
    # --------------------------------------------------------

    hand_data = extract_hand_data(
        hand_landmarks,
        width,
        height
    )

    features.update(
        hand_data
    )

    # --------------------------------------------------------
    # Head pose
    # --------------------------------------------------------

    head_data = estimate_head_pose(
        face_landmarks,
        width,
        height
    )

    features.update(
        head_data
    )

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

    object_data = object_feature_data(
        detections
    )

    features.update(
        object_data
    )

    # --------------------------------------------------------
    # Gaze
    # --------------------------------------------------------

    gaze_result = run_gazenet(
        crop_face_from_landmarks(
            frame,
            face_landmarks
        )
    )

    gaze = gaze_result[
        "gaze"
    ]

    gaze_direction = (
        gaze_direction_from_vector(
            gaze
        )
    )

    features[
        "gaze_direction"
    ] = gaze_direction

    screen = gaze_result[
        "screen"
    ]

    features[
        "gazePoint_x"
    ] = float(
        screen[0]
    )

    features[
        "gazePoint_y"
    ] = float(
        screen[1]
    )

    features[
        "gaze_on_script"
    ] = int(
        gaze_direction
        == "center"
    )

    # --------------------------------------------------------
    # Pupils
    # --------------------------------------------------------

    features.update(
        pupil_features(
            face_landmarks,
            width,
            height
        )
    )

    # --------------------------------------------------------
    # Exact categorical defaults
    # --------------------------------------------------------

    if not features[
        "face_present"
    ]:

        features[
            "head_pose"
        ] = "center"

        features[
            "gaze_direction"
        ] = "center"

    # --------------------------------------------------------
    # Ensure exact 37 features
    # --------------------------------------------------------

    return {
        name: features.get(
            name,
            0
        )
        for name in feature_names
    }


# ============================================================
# XGBOOST CLASSIFICATION
# ============================================================

def classify_behavior(
    features
):

    if xgboost_engine is None:

        return {
            "prediction": "Normal",
            "label": 0,
            "confidence": 0.0,
            "normal_probability": 0.0,
            "cheating_probability": 0.0,
            "features_ready": False,
            "reasons": [
                "XGBoost not ready"
            ],
        }

    try:

        return (
            xgboost_engine.predict_safe(
                features
            )
        )

    except Exception as exc:

        print(
            "XGBoost classification error:",
            exc
        )

        return {
            "prediction": "Normal",
            "label": 0,
            "confidence": 0.0,
            "normal_probability": 0.0,
            "cheating_probability": 0.0,
            "features_ready": False,
            "reasons": [
                str(exc)
            ],
        }


# ============================================================
# CONFIG STATE UPDATE
# ============================================================

def update_config_state(
    detections,
    features,
    behavior,
    identity
):

    config.DETECTED_OBJECTS = [
        normalize_label(
            d.get(
                "label",
                ""
            )
        )
        for d in detections
    ]

    for class_name in (
        config.YOLO11M_CLASSES
    ):

        key = class_name

        config.OBJECT_STATE[
            key
        ]["present"] = False

    for detection in detections:

        label = detection.get(
            "label",
            ""
        )

        matched = None

        for class_name in (
            config.YOLO11M_CLASSES
        ):

            if normalize_label(
                class_name
            ) == normalize_label(
                label
            ):

                matched = class_name
                break

        if matched is None:
            continue

        x1, y1, x2, y2 = (
            detection["box"]
        )

        config.OBJECT_STATE[
            matched
        ] = {
            "present": True,
            "confidence": float(
                detection.get(
                    "confidence",
                    0.0
                )
            ),
            "x": float(x1),
            "y": float(y1),
            "w": float(
                x2 - x1
            ),
            "h": float(
                y2 - y1
            ),
        }

    config.DETECTION_STATE.update(
        features
    )

    config.IDENTITY_STATE.update(
        {
            "verified": bool(
                identity.get(
                    "verified",
                    False
                )
            ),
            "confidence": float(
                identity.get(
                    "similarity",
                    0.0
                )
            ),
            "student_id": config.STUDENT_ID,
            "student_name": identity.get(
                "identity",
                "UNKNOWN"
            ),
        }
    )

    config.BEHAVIOR_STATE.update(
        behavior
    )

    config.SYSTEM_STATE[
        "processed_frames"
    ] += 1

    config.SYSTEM_STATE[
        "last_frame_time"
    ] = time.time()


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_overlay(
    frame,
    detections,
    features,
    behavior,
    identity
):

    output = frame.copy()

    # --------------------------------------------------------
    # YOLO boxes
    # --------------------------------------------------------

    for detection in detections:

        x1, y1, x2, y2 = (
            map(
                int,
                detection["box"]
            )
        )

        label = str(
            detection.get(
                "label",
                ""
            )
        )

        confidence = float(
            detection.get(
                "confidence",
                0.0
            )
        )

        text = (
            f"{label} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            output,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            text,
            (
                x1,
                max(
                    20,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    behavior_name = behavior.get(
        "prediction",
        "Normal"
    )

    behavior_confidence = float(
        behavior.get(
            "confidence",
            0.0
        )
    )

    identity_status = (
        "VERIFIED"
        if identity.get(
            "verified",
            False
        )
        else
        "NOT VERIFIED"
    )

    gaze_direction = features.get(
        "gaze_direction",
        "center"
    )

    head_pose = features.get(
        "head_pose",
        "center"
    )

    lines = [
        f"Behavior: {behavior_name} "
        f"{behavior_confidence:.2f}",

        f"Identity: {identity_status}",

        f"Gaze: {gaze_direction}",

        f"Head: {head_pose}",
    ]

    y = 30

    for line in lines:

        cv2.putText(
            output,
            line,
            (
                20,
                y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 28

    return output


# ============================================================
# INITIALIZE ALL AI COMPONENTS
# ============================================================

def initialize():

    print()
    print("=" * 70)
    print("AI EXAM SURVEILLANCE - INITIALIZATION")
    print("=" * 70)

    yolo_ready = (
        load_yolo_model()
    )

    mediapipe_ready = (
        initialize_mediapipe()
    )

    gaze_ready = (
        load_gazenet()
    )

    face_ready = (
        initialize_face_identity()
    )

    xgb_ready = (
        initialize_xgboost()
    )

    config.SYSTEM_STATE[
        "ai_pipeline"
    ].update(
        {
            "yolo11m": yolo_ready,
            "mediapipe": mediapipe_ready,
            "gazenet_v2": gaze_ready,
            "face_identity": face_ready,
            "xgboost": xgb_ready,
        }
    )

    print()
    print(
        "YOLO11m     :",
        "READY" if yolo_ready else "NOT READY"
    )

    print(
        "MediaPipe   :",
        "READY" if mediapipe_ready else "NOT READY"
    )

    print(
        "GazeNet V2  :",
        "READY" if gaze_ready else "NOT READY"
    )

    print(
        "Face ID     :",
        "READY" if face_ready else "NOT READY"
    )

    print(
        "XGBoost     :",
        "READY" if xgb_ready else "NOT READY"
    )

    print("=" * 70)

    return {
        "yolo11m": yolo_ready,
        "mediapipe": mediapipe_ready,
        "gazenet_v2": gaze_ready,
        "face_identity": face_ready,
        "xgboost": xgb_ready,
    }


# ============================================================
# MAIN DETECTION FUNCTION
# ============================================================

def detect(
    frame
):

    global _fps_counter
    global _fps_start_time

    if frame is None:

        return frame

    start_time = time.time()

    try:

        # ----------------------------------------------------
        # YOLO11m
        # ----------------------------------------------------

        detections = run_yolo(
            frame
        )

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        mediapipe_data = (
            run_mediapipe(
                frame
            )
        )

        # ----------------------------------------------------
        # Convert hand landmarks to temporary detections
        # for feature interaction calculations.
        # ----------------------------------------------------

        hand_detections = []

        height, width = (
            frame.shape[:2]
        )

        for hand in (
            mediapipe_data[
                "hand_landmarks"
            ]
        ):

            try:

                xs = [
                    p.x * width
                    for p in hand.landmark
                ]

                ys = [
                    p.y * height
                    for p in hand.landmark
                ]

                box = (
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys)
                )

                hand_detections.append(
                    {
                        "type": "hand",
                        "label": "hand",
                        "box": box,
                    }
                )

            except Exception:
                continue

        combined_detections = (
            detections
            +
            hand_detections
        )

        # ----------------------------------------------------
        # Face crop
        # ----------------------------------------------------

        face_crop = (
            crop_face_from_landmarks(
                frame,
                mediapipe_data[
                    "face_landmarks"
                ]
            )
        )

        # ----------------------------------------------------
        # Face identity
        # ----------------------------------------------------

        identity = (
            run_face_identity(
                face_crop
            )
        )

        # ----------------------------------------------------
        # Exact 37 features
        # ----------------------------------------------------

        features = (
            build_xgboost_features(
                combined_detections,
                frame,
                mediapipe_data
            )
        )

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        behavior = (
            classify_behavior(
                features
            )
        )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        update_config_state(
            detections,
            features,
            behavior,
            identity
        )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        _fps_counter += 1

        now = time.time()

        elapsed = (
            now - _fps_start_time
        )

        if elapsed >= 1.0:

            config.SYSTEM_STATE[
                "fps"
            ] = (
                _fps_counter
                /
                elapsed
            )

            _fps_counter = 0

            _fps_start_time = now

        # ----------------------------------------------------
        # Inference timing
        # ----------------------------------------------------

        config.SYSTEM_STATE[
            "last_inference_ms"
        ] = (
            time.time()
            -
            start_time
        ) * 1000.0

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        processed = (
            draw_overlay(
                frame,
                detections,
                features,
                behavior,
                identity
            )
        )

        return processed

    except Exception as exc:

        print(
            "Detector pipeline error:",
            exc
        )

        return frame


# ============================================================
# INITIALIZE WHEN IMPORTED
# ============================================================

try:

    initialize()

except Exception as exc:

    print(
        "Detector initialization error:",
        exc
    )


# ============================================================
# STATUS
# ============================================================

def get_status():

    return {
        "yolo11m": yolo_model is not None,
        "mediapipe": (
            face_mesh is not None
            or
            hands_detector is not None
        ),
        "gazenet_v2": gaze_model is not None,
        "face_identity": (
            embedding_model is not None
            and
            reference_embedding is not None
        ),
        "xgboost": (
            xgboost_engine is not None
            and
            xgboost_engine.loaded
        ),
        "device": (
            str(DEVICE)
            if DEVICE is not None
            else "unavailable"
        ),
        "feature_count": len(
            config.FEATURE_NAMES
        ),
    }


# ============================================================
# SHUTDOWN
# ============================================================

def shutdown():

    global face_mesh
    global hands_detector

    try:

        if face_mesh is not None:
            face_mesh.close()

    except Exception:
        pass

    try:

        if hands_detector is not None:
            hands_detector.close()

    except Exception:
        pass

    face_mesh = None
    hands_detector = None


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("AI EXAM SURVEILLANCE - DETECTOR TEST")
    print("=" * 70)

    print(
        get_status()
    )

    print()
    print(
        "Expected feature count:",
        37
    )

    print(
        "Configured feature count:",
        len(
            config.FEATURE_NAMES
        )
    )

    print("=" * 70)