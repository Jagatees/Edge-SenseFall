import os
import time
import threading
import smtplib
from collections import deque
from datetime import datetime
from email.message import EmailMessage
from alert_service import send_fall_alert

import cv2
import requests
from flask import Flask, jsonify, render_template_string, Response

from sensors.camera import Camera
from inference.pose_detection import PoseEstimator


app = Flask(__name__)
# DEFAULT_ALERT_EMAIL = "jagateesvaran@gmail.com"


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.last_frame_ts = None
        self.person_visible = False
        self.last_fall_ts = None
        self.total_fall_events = 0
        self.last_error = None
        self.events = deque(maxlen=100)
        self.latest_frame = None
        self.camera_connected = False
        self.last_no_frame_log_ts = 0

        # self.last_email_sent_ts = 0
        # self.email_cooldown_sec = int(os.getenv("EMAIL_COOLDOWN_SEC", "60"))

    def to_dict(self):
        with self.lock:
            return {
                "running": self.running,
                "last_frame_ts": self.last_frame_ts,
                "person_visible": self.person_visible,
                "camera_connected": self.camera_connected,
                "last_fall_ts": self.last_fall_ts,
                "total_fall_events": self.total_fall_events,
                "last_error": self.last_error,
                "events": list(self.events),
            }

    def add_event(self, level, message):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            self.events.appendleft({"time": ts, "level": level, "message": message})


state = AppState()


def _resolve_cloud_events_url():
    base = os.getenv("CLOUD_SYNC_URL", "").strip()
    if not base:
        return None
    if base.endswith("/api/events"):
        return base
    return f"{base.rstrip('/')}/api/events"


def sync_event_to_cloud(event_time, event_type, confidence=None, metadata=None):
    url = _resolve_cloud_events_url()
    if not url:
        return

    api_key = os.getenv("CLOUD_SYNC_API_KEY", "").strip()
    device_id = os.getenv("CLOUD_DEVICE_ID", "home_pi_01")
    timeout_sec = float(os.getenv("CLOUD_SYNC_TIMEOUT_SEC", "5"))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    payload = {
        "event_time": event_time,
        "event_type": event_type,
        "confidence": confidence,
        "device_id": device_id,
        "source": "raspberry-pi",
        "metadata": metadata or {},
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
        if resp.status_code >= 400:
            state.add_event("WARN", f"Cloud sync failed: HTTP {resp.status_code}")
        else:
            state.add_event("INFO", "Cloud sync successful")
    except Exception as exc:
        state.add_event("WARN", f"Cloud sync error: {exc}")


# def send_fall_email_alert(event_time, is_test=False):
#     smtp_host = os.getenv("ALERT_SMTP_HOST")
#     smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
#     smtp_user = os.getenv("ALERT_SMTP_USER")
#     smtp_pass = os.getenv("ALERT_SMTP_PASS")
#     sender = os.getenv("ALERT_FROM_EMAIL", DEFAULT_ALERT_EMAIL)
#     recipient = os.getenv("ALERT_TO_EMAIL", DEFAULT_ALERT_EMAIL)

#     missing = []
#     if not smtp_host:
#         missing.append("ALERT_SMTP_HOST")
#     if not smtp_user:
#         missing.append("ALERT_SMTP_USER")
#     if not smtp_pass:
#         missing.append("ALERT_SMTP_PASS")
#     if not sender:
#         missing.append("ALERT_FROM_EMAIL")
#     if not recipient:
#         missing.append("ALERT_TO_EMAIL")

#     if missing:
#         state.add_event("WARN", f"Email not sent: missing {', '.join(missing)}")
#         return

#     now = time.time()
#     if not is_test:
#         with state.lock:
#             if now - state.last_email_sent_ts < state.email_cooldown_sec:
#                 state.add_event("INFO", "Email skipped due to cooldown")
#                 return

#     msg = EmailMessage()
#     if is_test:
#         msg["Subject"] = "SmartFall Test Email"
#     else:
#         msg["Subject"] = "SmartFall Alert: Possible Fall Detected"
#     msg["From"] = sender
#     msg["To"] = recipient
#     if is_test:
#         msg.set_content(
#             f"This is a test email from SmartFall dashboard at {event_time}.\n"
#             "SMTP configuration is working."
#         )
#     else:
#         msg.set_content(
#             f"A possible fall was detected at {event_time}.\n"
#             "Please check on the person immediately."
#         )

#     try:
#         with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
#             server.starttls()
#             server.login(smtp_user, smtp_pass)
#             server.send_message(msg)

#         if not is_test:
#             with state.lock:
#                 state.last_email_sent_ts = now
#             state.add_event("INFO", f"Email alert sent to {recipient}")
#         else:
#             state.add_event("INFO", f"Test email sent to {recipient}")
#     except Exception as exc:
#         with state.lock:
#             state.last_error = f"Email error: {exc}"
#         state.add_event("ERROR", f"Failed to send email: {exc}")


def detector_loop():
    camera = Camera()
    pose_model = PoseEstimator()
    state.add_event("INFO", "Detection loop started")

    while True:
        with state.lock:
            if not state.running:
                break

        try:
            frame = camera.get_frame()
            with state.lock:
                state.last_frame_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                state.latest_frame = frame

            if frame is None:
                now = time.time()
                with state.lock:
                    state.camera_connected = False
                    should_log = now - state.last_no_frame_log_ts >= 5
                    if should_log:
                        state.last_no_frame_log_ts = now
                if should_log:
                    state.add_event("WARN", "Camera not connected or no frames yet")
                time.sleep(0.2)
                continue
            else:
                with state.lock:
                    state.camera_connected = True

            person_visible = camera.detect_person(frame)
            with state.lock:
                state.person_visible = person_visible

            if not person_visible:
                time.sleep(0.2)
                continue

            frame_height, frame_width, _ = frame.shape
            keypoints = pose_model.estimate_pose(frame)
            camera_fall = pose_model.detect_fall_pose(keypoints, frame_width, frame_height)

            if camera_fall:
                event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with state.lock:
                    state.last_fall_ts = event_time
                    state.total_fall_events += 1
                state.add_event("ALERT", f"Possible fall detected at {event_time}")
                sync_event_to_cloud(
                    event_time=event_time,
                    event_type="fall",
                    confidence=0.95,
                    metadata={"sensor": "camera_pose"},
                )
                success = send_fall_alert(
                    confidence=0.9,
                    trigger_source="camera_pose",
                    location="home"
                )
                
                state.add_event("INFO", "Fall alert email sent")
                

            time.sleep(0.2)
        except Exception as exc:
            with state.lock:
                state.last_error = str(exc)
            state.add_event("ERROR", f"Detection loop error: {exc}")
            time.sleep(0.5)


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SmartFall Dashboard</title>
  <style>
    :root {
      --bg: #f3f6fb;
      --card: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --ok: #0f766e;
      --warn: #b45309;
      --alert: #b91c1c;
      --line: #e5e7eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: radial-gradient(circle at top right, #dbeafe 0%, var(--bg) 45%);
      color: var(--ink);
    }
    .wrap {
      max-width: 980px;
      margin: 24px auto;
      padding: 0 16px;
      display: grid;
      gap: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 6px 18px rgba(31, 41, 55, 0.06);
    }
    h1 { margin: 0 0 8px 0; font-size: 1.6rem; }
    .subtitle { color: var(--muted); margin: 0; }
    .stats {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    .stat { border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
    .label { color: var(--muted); font-size: 0.85rem; }
    .value { margin-top: 6px; font-size: 1.1rem; font-weight: 700; }
    .pill { display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 0.8rem; }
    .ok { background: #ccfbf1; color: var(--ok); }
    .warn { background: #ffedd5; color: var(--warn); }
    .alert { background: #fee2e2; color: var(--alert); }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 10px; border-bottom: 1px solid var(--line); font-size: 0.92rem; }
    th { color: var(--muted); font-weight: 600; }
    .footer { color: var(--muted); font-size: 0.85rem; }
    .actions { display: flex; gap: 10px; align-items: center; }
    .btn {
      border: 1px solid #0ea5e9;
      background: #e0f2fe;
      color: #075985;
      border-radius: 10px;
      padding: 8px 12px;
      font-weight: 600;
      cursor: pointer;
    }
    .btn:disabled { opacity: 0.6; cursor: not-allowed; }
    .status { color: var(--muted); font-size: 0.9rem; }
    .preview {
      width: 100%;
      max-width: 640px;
      border: 1px solid var(--line);
      border-radius: 10px;
      display: block;
      background: #111827;
    }
    .previewWrap { position: relative; max-width: 640px; }
    .previewMsg {
      position: absolute;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      color: #e5e7eb;
      background: #111827;
      border: 1px dashed #374151;
      border-radius: 10px;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>SmartFall Live Dashboard</h1>
      <p class="subtitle">One-page monitor for fall detection events and alert status</p>
    </div>

    <div class="card stats">
      <div class="stat"><div class="label">System</div><div class="value" id="running">-</div></div>
      <div class="stat"><div class="label">Camera</div><div class="value" id="camera">-</div></div>
      <div class="stat"><div class="label">Person Visible</div><div class="value" id="person">-</div></div>
      <div class="stat"><div class="label">Total Fall Events</div><div class="value" id="total">-</div></div>
      <div class="stat"><div class="label">Last Fall</div><div class="value" id="lastFall">-</div></div>
      <div class="stat"><div class="label">Last Frame</div><div class="value" id="lastFrame">-</div></div>
      <div class="stat"><div class="label">Last Error</div><div class="value" id="lastError">-</div></div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Live Camera Preview</h3>
      <div class="previewWrap">
        <img id="preview" class="preview" src="/api/frame" alt="Live camera frame" />
        <div id="previewMsg" class="previewMsg">Waiting for camera connection...</div>
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Alerts</h3>
      <div class="actions">
        <button id="testEmailBtn" class="btn">Send Test Email</button>
        <span id="testEmailStatus" class="status">Idle</span>
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Event Log</h3>
      <table>
        <thead>
          <tr><th>Time</th><th>Level</th><th>Message</th></tr>
        </thead>
        <tbody id="events"></tbody>
      </table>
    </div>

    <div class="footer">Auto-refresh every 2 seconds.</div>
  </div>

<script>
  function levelPill(level) {
    const l = (level || '').toUpperCase();
    if (l === 'ALERT' || l === 'ERROR') return '<span class="pill alert">' + l + '</span>';
    if (l === 'WARN') return '<span class="pill warn">' + l + '</span>';
    return '<span class="pill ok">' + l + '</span>';
  }

  async function refresh() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();

      document.getElementById('running').textContent = data.running ? 'Running' : 'Stopped';
      document.getElementById('camera').textContent = data.camera_connected ? 'Connected' : 'Waiting';
      document.getElementById('person').textContent = data.person_visible ? 'Yes' : 'No';
      document.getElementById('total').textContent = String(data.total_fall_events || 0);
      document.getElementById('lastFall').textContent = data.last_fall_ts || '-';
      document.getElementById('lastFrame').textContent = data.last_frame_ts || '-';
      document.getElementById('lastError').textContent = data.last_error || '-';

      const events = document.getElementById('events');
      events.innerHTML = '';
      (data.events || []).forEach(e => {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td>' + (e.time || '-') + '</td>' +
                       '<td>' + levelPill(e.level) + '</td>' +
                       '<td>' + (e.message || '-') + '</td>';
        events.appendChild(tr);
      });
    } catch (err) {
      document.getElementById('lastError').textContent = 'Dashboard fetch error: ' + err;
    }
  }

  function updatePreview() {
    const img = document.getElementById('preview');
    const msg = document.getElementById('previewMsg');
    img.onload = () => { msg.style.display = 'none'; };
    img.onerror = () => { msg.style.display = 'flex'; };
    img.src = '/api/frame?ts=' + Date.now();
  }

  async function sendTestEmail() {
    const btn = document.getElementById('testEmailBtn');
    const status = document.getElementById('testEmailStatus');
    btn.disabled = true;
    status.textContent = 'Sending...';
    try {
      const res = await fetch('/api/test-email', { method: 'POST' });
      const data = await res.json();
      status.textContent = data.message || 'Done';
      await refresh();
    } catch (err) {
      status.textContent = 'Failed: ' + err;
    } finally {
      btn.disabled = false;
    }
  }

  document.getElementById('testEmailBtn').addEventListener('click', sendTestEmail);
  refresh();
  updatePreview();
  setInterval(refresh, 2000);
  setInterval(updatePreview, 2000);
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/status")
def api_status():
    return jsonify(state.to_dict())


@app.route("/api/frame")
def api_frame():
    with state.lock:
        frame = None if state.latest_frame is None else state.latest_frame.copy()
    if frame is None:
        return Response(status=204)

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return Response(status=500)
    return Response(encoded.tobytes(), mimetype="image/jpeg")


@app.route("/api/test-email", methods=["POST"])
def api_test_email():
    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #send_fall_email_alert(event_time, is_test=True)
    # ✅ Simulate fall event
    with state.lock:
        state.last_fall_ts = event_time
        state.total_fall_events += 1

    state.add_event("ALERT", f"[TEST] Simulated fall at {event_time}")

    # ✅ Trigger your alert_service email
    send_fall_alert(
        confidence=0.95,
        trigger_source="manual test",
        location="dashboard"
    )

    state.add_event("INFO", "Test fall triggered (email sent)")
    return jsonify({"ok": True, "message": "Test email request processed. Check event log."})


if __name__ == "__main__":
    detector_thread = threading.Thread(target=detector_loop, daemon=True)
    detector_thread.start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
