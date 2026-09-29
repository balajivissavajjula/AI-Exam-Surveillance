"""
AI Exam Surveillance - Flask Application
"""

import atexit
import logging
from datetime import datetime

from flask import Flask, jsonify, render_template, Response

import config
import camera
import detector


# ============================================================
# APPLICATION
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("AI-Exam-Surveillance")


# ============================================================
# AI PIPELINE
# ============================================================

AI_PIPELINE = {
    "camera": "OpenCV",
    "object_detection": "YOLO11m",
    "landmarks_head_pose": "MediaPipe",
    "gaze_estimation": "GazeNet V2",
    "identity_verification": "Face Identity Module",
    "feature_extraction": "37-feature representation",
    "behavior_classifier": "XGBoost",
    "behavior_classes": [
        "Normal",
        "Cheating"
    ],
    "cloud_logging": "Firebase",
    "iot_alert": "ESP32"
}


YOLO11M_CLASSES = [
    "phone",
    "book",
    "cheat_paper",
    "calculator",
    "earphone",
    "sunglasses",
    "watch",
    "Answer_paper",
    "laptop"
]


# ============================================================
# APPLICATION STATE
# ============================================================

APPLICATION_STATE = {
    "exam_active": False,
    "exam_start_time": None,
    "exam_stop_time": None,
    "last_update": None
}


# ============================================================
# DETECTOR
# ============================================================

def get_detector_status():

    try:

        if hasattr(detector, "get_status"):

            result = detector.get_status()

            if isinstance(result, dict):
                return result

            return {
                "status": str(result)
            }

    except Exception as exc:

        logger.error(
            "Detector status error: %s",
            exc
        )

    return {
        "status": "unknown"
    }


def initialize_detector():

    try:

        if hasattr(detector, "initialize"):

            result = detector.initialize()

            logger.info(
                "Detector initialized."
            )

            return {
                "success": True,
                "result": result
            }

        return {
            "success": True,
            "result": "Detector initialization handled internally."
        }

    except Exception as exc:

        logger.exception(
            "Detector initialization failed."
        )

        return {
            "success": False,
            "error": str(exc)
        }


def shutdown_detector():

    try:

        if hasattr(detector, "shutdown"):
            detector.shutdown()

    except Exception:

        logger.exception(
            "Detector shutdown error."
        )


# ============================================================
# CAMERA
# ============================================================

def get_camera_object():

    for name in (
        "camera",
        "Camera",
        "camera_instance",
        "CAMERA"
    ):

        if hasattr(camera, name):

            obj = getattr(camera, name)

            if obj is not None:
                return obj

    return None


def generate_video_stream():

    cam = get_camera_object()

    if cam is None:

        logger.error(
            "Camera object not available."
        )

        return

    try:

        if hasattr(cam, "generate_frames"):

            yield from cam.generate_frames()
            return

        if hasattr(cam, "get_frame"):

            while True:

                frame = cam.get_frame()

                if frame is None:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

    except GeneratorExit:

        return

    except Exception:

        logger.exception(
            "Video streaming error."
        )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    try:

        return render_template(
            "index.html",
            pipeline=AI_PIPELINE,
            classes=YOLO11M_CLASSES,
            state=APPLICATION_STATE
        )

    except Exception:

        return jsonify({
            "project": "AI Exam Surveillance",
            "status": "running",
            "pipeline": AI_PIPELINE,
            "classes": YOLO11M_CLASSES,
            "exam_active": APPLICATION_STATE["exam_active"]
        })


# ============================================================
# VIDEO
# ============================================================

@app.route("/video")
def video():

    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard_data")
def dashboard_data():

    detector_status = get_detector_status()

    APPLICATION_STATE["last_update"] = (
        datetime.now().isoformat()
    )

    return jsonify({

        "success": True,

        "timestamp":
            APPLICATION_STATE["last_update"],

        "exam": {
            "active":
                APPLICATION_STATE["exam_active"],

            "start_time":
                APPLICATION_STATE["exam_start_time"],

            "stop_time":
                APPLICATION_STATE["exam_stop_time"]
        },

        "pipeline":
            AI_PIPELINE,

        "yolo11m": {
            "classes":
                YOLO11M_CLASSES
        },

        "detector":
            detector_status
    })


# ============================================================
# START EXAM
# ============================================================

@app.route(
    "/start_exam",
    methods=["GET", "POST"]
)
def start_exam():

    APPLICATION_STATE["exam_active"] = True

    APPLICATION_STATE["exam_start_time"] = (
        datetime.now().isoformat()
    )

    APPLICATION_STATE["exam_stop_time"] = None

    logger.info(
        "Exam started."
    )

    return jsonify({

        "success": True,

        "message":
            "Exam started.",

        "exam_active":
            True,

        "start_time":
            APPLICATION_STATE["exam_start_time"]
    })


# ============================================================
# STOP EXAM
# ============================================================

@app.route(
    "/stop_exam",
    methods=["GET", "POST"]
)
def stop_exam():

    APPLICATION_STATE["exam_active"] = False

    APPLICATION_STATE["exam_stop_time"] = (
        datetime.now().isoformat()
    )

    logger.info(
        "Exam stopped."
    )

    return jsonify({

        "success": True,

        "message":
            "Exam stopped.",

        "exam_active":
            False,

        "stop_time":
            APPLICATION_STATE["exam_stop_time"]
    })


# ============================================================
# RESET
# ============================================================

@app.route(
    "/reset",
    methods=["GET", "POST"]
)
def reset():

    APPLICATION_STATE["exam_active"] = False

    APPLICATION_STATE["exam_start_time"] = None

    APPLICATION_STATE["exam_stop_time"] = None

    try:

        if hasattr(
            config,
            "reset_detection_state"
        ):

            config.reset_detection_state()

    except Exception:

        logger.exception(
            "Configuration reset failed."
        )

    try:

        if hasattr(
            detector,
            "reset"
        ):

            detector.reset()

    except Exception:

        logger.exception(
            "Detector reset failed."
        )

    return jsonify({

        "success": True,

        "message":
            "System reset successfully."
    })


# ============================================================
# CAMERA STATUS
# ============================================================

@app.route("/camera_status")
def camera_status():

    cam = get_camera_object()

    if cam is None:

        return jsonify({

            "success": False,

            "camera_available":
                False,

            "message":
                "Camera object not available."
        })

    status = {}

    try:

        if hasattr(
            cam,
            "get_status"
        ):

            result = cam.get_status()

            if isinstance(
                result,
                dict
            ):

                status.update(result)

        elif hasattr(
            cam,
            "status"
        ):

            result = cam.status

            if isinstance(
                result,
                dict
            ):

                status.update(result)

            else:

                status["status"] = str(result)

    except Exception as exc:

        status["error"] = str(exc)

    status.setdefault(
        "camera_available",
        True
    )

    return jsonify({

        "success": True,

        "camera":
            status
    })


# ============================================================
# SYSTEM STATUS
# ============================================================

@app.route("/status")
def status():

    return jsonify({

        "success": True,

        "application":
            "AI Exam Surveillance",

        "status":
            "running",

        "exam_active":
            APPLICATION_STATE["exam_active"],

        "pipeline":
            AI_PIPELINE,

        "detector":
            get_detector_status()
    })


# ============================================================
# EXAM CONFIG
# ============================================================

@app.route("/exam_config")
def exam_config():

    data = {

        "success": True,

        "classes":
            YOLO11M_CLASSES,

        "pipeline":
            AI_PIPELINE
    }

    try:

        if hasattr(
            config,
            "get_camera_source"
        ):

            data["camera_source"] = (
                config.get_camera_source()
            )

        elif hasattr(
            config,
            "CAMERA_SOURCE"
        ):

            data["camera_source"] = (
                config.CAMERA_SOURCE
            )

    except Exception:
        pass

    try:

        if hasattr(
            config,
            "RESTRICTED_ITEMS"
        ):

            data["restricted_items"] = (
                config.RESTRICTED_ITEMS
            )

    except Exception:
        pass

    return jsonify(data)


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "healthy",

        "service":
            "AI Exam Surveillance",

        "timestamp":
            datetime.now().isoformat(),

        "exam_active":
            APPLICATION_STATE["exam_active"]
    })


# ============================================================
# PIPELINE API
# ============================================================

@app.route("/api/pipeline")
def pipeline():

    return jsonify({

        "success": True,

        "pipeline":
            AI_PIPELINE,

        "yolo_classes":
            YOLO11M_CLASSES,

        "behavior_classes": [
            "Normal",
            "Cheating"
        ],

        "feature_count":
            37
    })


# ============================================================
# DETECTOR API
# ============================================================

@app.route("/api/detector")
def detector_info():

    return jsonify({

        "success": True,

        "status":
            get_detector_status()
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success": False,

        "error":
            "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):

    logger.exception(
        "Internal server error."
    )

    return jsonify({

        "success": False,

        "error":
            "Internal server error"
    }), 500


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_application():

    logger.info(
        "=" * 60
    )

    logger.info(
        "AI EXAM SURVEILLANCE"
    )

    logger.info(
        "=" * 60
    )

    logger.info(
        "Object Detection : YOLO11m"
    )

    logger.info(
        "Landmarks        : MediaPipe"
    )

    logger.info(
        "Gaze Estimation  : GazeNet V2"
    )

    logger.info(
        "Identity         : Face Identity Module"
    )

    logger.info(
        "Features         : 37"
    )

    logger.info(
        "Classifier       : XGBoost"
    )

    logger.info(
        "Cloud Logging    : Firebase"
    )

    logger.info(
        "IoT Alert        : ESP32"
    )

    logger.info(
        "=" * 60
    )

    result = initialize_detector()

    if not result["success"]:

        logger.warning(
            "Detector initialization error: %s",
            result.get("error")
        )


# ============================================================
# SHUTDOWN
# ============================================================

def shutdown_application():

    logger.info(
        "Shutting down AI Exam Surveillance..."
    )

    shutdown_detector()

    cam = get_camera_object()

    try:

        if (
            cam is not None
            and hasattr(
                cam,
                "shutdown"
            )
        ):

            cam.shutdown()

    except Exception:

        logger.exception(
            "Camera shutdown error."
        )

    logger.info(
        "Shutdown complete."
    )


atexit.register(
    shutdown_application
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    initialize_application()

    host = getattr(
        config,
        "HOST",
        "0.0.0.0"
    )

    port = int(
        getattr(
            config,
            "PORT",
            5000
        )
    )

    debug = bool(
        getattr(
            config,
            "DEBUG",
            False
        )
    )

    logger.info(
        "Starting Flask server at http://%s:%s",
        host,
        port
    )

    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False
    )
