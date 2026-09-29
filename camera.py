"""
AI Exam Surveillance
Camera Capture and MJPEG Streaming

Responsibilities:
    - OpenCV camera capture
    - Camera reconnect
    - Background frame capture
    - Frame buffering
    - Detector integration
    - MJPEG streaming
"""

import cv2
import time
import threading
import numpy as np

import config


# ============================================================
# CAMERA CLASS
# ============================================================

class Camera:

    def __init__(self, source=None):

        self.source = (
            config.get_camera_source()
            if source is None
            else source
        )

        self.cap = None

        self.running = False
        self.connected = False

        self.thread = None

        self.lock = threading.RLock()

        self.last_frame = None
        self.last_frame_time = 0.0

        self.frame_count = 0

        self.width = int(
            getattr(
                config,
                "CAMERA_WIDTH",
                1280
            )
        )

        self.height = int(
            getattr(
                config,
                "CAMERA_HEIGHT",
                720
            )
        )

        self.fps = int(
            getattr(
                config,
                "CAMERA_FPS",
                20
            )
        )

        self.reconnect_delay = float(
            getattr(
                config,
                "CAMERA_RECONNECT_DELAY",
                2.0
            )
        )

        self.buffer_size = int(
            getattr(
                config,
                "CAMERA_BUFFER_SIZE",
                1
            )
        )

        self._stop_event = threading.Event()


    # ========================================================
    # OPEN CAMERA
    # ========================================================

    def _open_capture(self):

        try:

            if self.cap is not None:

                try:
                    self.cap.release()
                except Exception:
                    pass

                self.cap = None

            self.cap = cv2.VideoCapture(
                self.source
            )

            if not self.cap.isOpened():

                self.connected = False

                try:
                    self.cap.release()
                except Exception:
                    pass

                self.cap = None

                return False

            # ------------------------------------------------
            # Camera properties
            # ------------------------------------------------

            try:
                self.cap.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    self.width
                )

                self.cap.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    self.height
                )

                self.cap.set(
                    cv2.CAP_PROP_FPS,
                    self.fps
                )

                self.cap.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    self.buffer_size
                )

            except Exception:
                pass

            self.connected = True

            return True

        except Exception as exc:

            print(
                "Camera open error:",
                exc
            )

            self.connected = False
            self.cap = None

            return False


    # ========================================================
    # START CAMERA
    # ========================================================

    def start(self):

        if self.running:
            return self.connected

        print(
            "Starting camera:",
            self.source
        )

        if not self._open_capture():

            print(
                "Camera connection failed."
            )

            return False

        self.running = True

        self._stop_event.clear()

        self.thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureThread",
            daemon=True
        )

        self.thread.start()

        return True


    # ========================================================
    # CAPTURE LOOP
    # ========================================================

    def _capture_loop(self):

        print(
            "Camera capture thread started."
        )

        while (
            self.running
            and not self._stop_event.is_set()
        ):

            try:

                if self.cap is None:

                    self.connected = False

                    time.sleep(
                        self.reconnect_delay
                    )

                    if self.running:

                        self._open_capture()

                    continue


                success, frame = (
                    self.cap.read()
                )


                if not success or frame is None:

                    self.connected = False

                    print(
                        "Camera frame read failed. "
                        "Attempting reconnect..."
                    )

                    try:
                        self.cap.release()
                    except Exception:
                        pass

                    self.cap = None

                    time.sleep(
                        self.reconnect_delay
                    )

                    if self.running:

                        self._open_capture()

                    continue


                # ------------------------------------------------
                # Store latest frame
                # ------------------------------------------------

                with self.lock:

                    self.last_frame = frame.copy()

                    self.last_frame_time = (
                        time.time()
                    )

                    self.frame_count += 1

                    self.connected = True


                # ------------------------------------------------
                # Small delay prevents unnecessary CPU usage
                # ------------------------------------------------

                if self.fps > 0:

                    time.sleep(
                        max(
                            0.001,
                            1.0 / self.fps
                        )
                    )

            except Exception as exc:

                print(
                    "Camera capture error:",
                    exc
                )

                self.connected = False

                time.sleep(0.10)


        print(
            "Camera capture thread stopped."
        )


    # ========================================================
    # GET FRAME
    # ========================================================

    def get_frame(self):

        with self.lock:

            if self.last_frame is None:
                return None

            return self.last_frame.copy()


    # ========================================================
    # CONNECTION STATUS
    # ========================================================

    def is_connected(self):

        return bool(
            self.connected
            and self.cap is not None
            and self.running
        )


    # ========================================================
    # CAMERA STATUS
    # ========================================================

    def get_status(self):

        with self.lock:

            return {
                "source": self.source,
                "running": bool(
                    self.running
                ),
                "connected": bool(
                    self.connected
                ),
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "frame_count": self.frame_count,
                "last_frame_time": (
                    self.last_frame_time
                ),
            }


    # ========================================================
    # STOP / RELEASE
    # ========================================================

    def release(self):

        print(
            "Releasing camera..."
        )

        self.running = False

        self._stop_event.set()

        thread = self.thread

        if thread is not None:

            try:

                if (
                    thread.is_alive()
                    and
                    thread
                    is not threading.current_thread()
                ):

                    thread.join(
                        timeout=2.0
                    )

            except Exception:
                pass

        with self.lock:

            if self.cap is not None:

                try:
                    self.cap.release()
                except Exception as exc:

                    print(
                        "Camera release error:",
                        exc
                    )

                self.cap = None

            self.connected = False

            self.last_frame = None

            self.last_frame_time = 0.0

        self.thread = None

        print(
            "Camera released."
        )


    # ========================================================
    # RECONNECT
    # ========================================================

    def reconnect(self):

        print(
            "Reconnecting camera..."
        )

        was_running = self.running

        self.running = False

        self._stop_event.set()

        thread = self.thread

        if thread is not None:

            try:

                if (
                    thread.is_alive()
                    and
                    thread
                    is not threading.current_thread()
                ):

                    thread.join(
                        timeout=2.0
                    )

            except Exception:
                pass

        with self.lock:

            if self.cap is not None:

                try:
                    self.cap.release()
                except Exception:
                    pass

                self.cap = None

            self.connected = False

        self.thread = None

        if was_running:

            time.sleep(
                0.2
            )

            return self.start()

        return False


# ============================================================
# GLOBAL CAMERA
# ============================================================

camera = Camera()


# ============================================================
# ERROR FRAME
# ============================================================

def create_camera_error_frame(
    message
):

    width = int(
        getattr(
            config,
            "CAMERA_WIDTH",
            1280
        )
    )

    height = int(
        getattr(
            config,
            "CAMERA_HEIGHT",
            720
        )
    )

    frame = np.full(
        (
            height,
            width,
            3
        ),
        30,
        dtype=np.uint8
    )

    cv2.putText(
        frame,
        str(message),
        (
            40,
            height // 2
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    try:

        cv2.putText(
            frame,
            f"Source: {camera.source}",
            (
                40,
                height // 2 + 40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

    except Exception:
        pass

    return frame


# ============================================================
# JPEG ENCODER
# ============================================================

def encode_frame(
    frame,
    quality=None
):

    try:

        if frame is None:
            return None

        if quality is None:

            quality = int(
                getattr(
                    config,
                    "CAMERA_JPEG_QUALITY",
                    85
                )
            )

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                int(quality),
            ]
        )

        if not success:
            return None

        return encoded.tobytes()

    except Exception as exc:

        print(
            "JPEG encode error:",
            exc
        )

        return None


# ============================================================
# MJPEG GENERATOR
# ============================================================

def generate_frames():

    try:

        from detector import detect

    except Exception as exc:

        print(
            "Detector import error:",
            exc
        )

        detect = None


    # --------------------------------------------------------
    # Start camera
    # --------------------------------------------------------

    if not camera.running:

        if not camera.start():

            print(
                "Initial camera start failed."
            )


    # --------------------------------------------------------
    # Streaming loop
    # --------------------------------------------------------

    while True:

        try:

            frame = camera.get_frame()


            # ------------------------------------------------
            # No camera frame
            # ------------------------------------------------

            if frame is None:

                blank = (
                    create_camera_error_frame(
                        "WAITING FOR CAMERA..."
                    )
                )

                encoded = encode_frame(
                    blank,
                    quality=75
                )

                if encoded is not None:

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n"
                        b"Pragma: no-cache\r\n\r\n"
                        + encoded
                        + b"\r\n"
                    )

                time.sleep(
                    0.05
                )

                continue


            # ------------------------------------------------
            # AI detector
            # ------------------------------------------------

            processed_frame = frame

            if detect is not None:

                try:

                    result = detect(
                        frame
                    )

                    if (
                        isinstance(
                            result,
                            tuple
                        )
                        and
                        len(result) >= 1
                    ):

                        processed_frame = (
                            result[0]
                        )

                    elif result is not None:

                        processed_frame = (
                            result
                        )

                except Exception as exc:

                    print(
                        "Detection error:",
                        exc
                    )

                    processed_frame = frame


            # ------------------------------------------------
            # Encode
            # ------------------------------------------------

            encoded = encode_frame(
                processed_frame
            )

            if encoded is None:

                time.sleep(
                    0.01
                )

                continue


            # ------------------------------------------------
            # MJPEG
            # ------------------------------------------------

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n"
                b"Pragma: no-cache\r\n\r\n"
                + encoded
                + b"\r\n"
            )

            time.sleep(
                0.01
            )


        except GeneratorExit:

            print(
                "Video client disconnected."
            )

            break


        except Exception as exc:

            print(
                "Video stream error:",
                exc
            )

            time.sleep(
                0.10
            )


# ============================================================
# SHUTDOWN
# ============================================================

def shutdown_camera():

    try:

        camera.release()

    except Exception as exc:

        print(
            "Camera shutdown error:",
            exc
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("AI EXAM SURVEILLANCE - CAMERA TEST")
    print("=" * 60)

    print(
        "Source:",
        camera.source
    )

    print(
        "Width:",
        camera.width
    )

    print(
        "Height:",
        camera.height
    )

    print(
        "FPS:",
        camera.fps
    )

    print()

    if camera.start():

        print(
            "Camera started."
        )

        print(
            "Press Q to exit."
        )

        try:

            while True:

                frame = camera.get_frame()

                if frame is not None:

                    cv2.imshow(
                        "AI Exam Camera Test",
                        frame
                    )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):
                    break

        except KeyboardInterrupt:

            pass

        finally:

            camera.release()

            cv2.destroyAllWindows()

    else:

        print(
            "Camera test FAILED."
        )

    print()

    print(
        "Camera status:"
    )

    print(
        camera.get_status()
    )

    print()

    print("=" * 60)