from collections import deque
from datetime import datetime
from pathlib import Path
import socket
import time

import cv2
from flask import Flask, Response, jsonify


app = Flask(__name__)
camera = cv2.VideoCapture(0)
boot_time = time.time()
snapshot_failures = 0
last_snapshot_iso = ""
last_alert_image = ""
archive_dir = Path.home() / "camera_archive"
archive_dir.mkdir(parents=True, exist_ok=True)
archive_index = deque(maxlen=20)


def local_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "Unavailable"


def uptime_string():
    seconds = max(0, int(time.time() - boot_time))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s"


def capture_frame():
    global snapshot_failures, last_snapshot_iso
    success, frame = camera.read()
    if not success:
        snapshot_failures += 1
        return None, None
    capture_iso = datetime.now().isoformat(timespec="seconds")
    last_snapshot_iso = capture_iso
    return frame, capture_iso


def archive_snapshot(frame, capture_iso, kind="snapshot"):
    filename = f"{kind}_{capture_iso.replace(':', '-').replace('T', '_')}.jpg"
    file_path = archive_dir / filename
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return ""
    file_path.write_bytes(encoded.tobytes())
    archive_index.appendleft(
        {
            "url": f"/archive/{filename}",
            "timestamp": capture_iso,
            "type": kind,
        }
    )
    return f"/archive/{filename}"


def jpeg_response(frame, capture_iso):
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return Response("Encode failed", status=500)
    response = Response(encoded.tobytes(), mimetype="image/jpeg")
    response.headers["X-Capture-Time"] = capture_iso
    response.headers["X-Pi-Uptime"] = uptime_string()
    response.headers["X-Pi-Local-IP"] = local_ip()
    response.headers["X-Tunnel-Updated-At"] = capture_iso
    response.headers["X-Camera-Guidance"] = "Check that the bed or monitored area is fully visible."
    response.headers["X-Last-Alert-Image"] = last_alert_image or "No alert snapshot"
    return response


@app.route("/")
def home():
    return '<h2>Pi Camera</h2><img src="/video_feed" style="max-width:100%;">'


@app.route("/video_feed")
def video_feed():
    def gen_frames():
        while True:
            frame, capture_iso = capture_frame()
            if frame is None:
                continue
            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"Captured: {capture_iso}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            ok, buffer = cv2.imencode(".jpg", overlay)
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot")
def snapshot():
    frame, capture_iso = capture_frame()
    if frame is None:
        return ("No frame", 503)
    overlay = frame.copy()
    cv2.putText(
        overlay,
        f"Captured: {capture_iso}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    archive_snapshot(overlay, capture_iso, kind="snapshot")
    return jpeg_response(overlay, capture_iso)


@app.route("/health.json")
def health():
    return jsonify(
        {
            "camera_online": True,
            "last_snapshot_time": last_snapshot_iso,
            "snapshot_failures": snapshot_failures,
            "pi_uptime": uptime_string(),
            "pi_local_ip": local_ip(),
            "last_alert_image": last_alert_image,
            "guidance": "Check that the bed or monitored area is fully visible.",
        }
    )


@app.route("/archive.json")
def archive_json():
    return jsonify({"items": list(archive_index)})


@app.route("/save_alert_snapshot")
def save_alert_snapshot():
    global last_alert_image
    frame, capture_iso = capture_frame()
    if frame is None:
        return ("No frame", 503)
    last_alert_image = archive_snapshot(frame, capture_iso, kind="alert")
    return jsonify({"ok": True, "image_url": last_alert_image, "capture_time": capture_iso})


@app.route("/archive/<path:filename>")
def archive_file(filename):
    file_path = archive_dir / filename
    if not file_path.exists():
        return ("Not found", 404)
    return Response(file_path.read_bytes(), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
