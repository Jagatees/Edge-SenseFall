import os
from flask import Flask, jsonify, request
import requests

app = Flask(__name__)


def _env(name, default=""):
    return os.getenv(name, default).strip()


def _supabase_headers(include_content_type=False):
    service_key = _env("SUPABASE_SERVICE_ROLE_KEY")
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
    }
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers


def _supabase_base_url():
    return f"{_env('SUPABASE_URL')}/rest/v1"


def _is_authorized(req):
    expected = _env("CLOUD_API_KEY")
    if not expected:
        return True
    provided = req.headers.get("x-api-key", "")
    return provided == expected


def _missing_config():
    missing = []
    if not _env("SUPABASE_URL"):
        missing.append("SUPABASE_URL")
    if not _env("SUPABASE_SERVICE_ROLE_KEY"):
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    return missing


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, x-api-key"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/health", methods=["GET"])
def health():
    missing = _missing_config()
    return jsonify({"ok": len(missing) == 0, "missing": missing})


@app.route("/api/events", methods=["OPTIONS"])
def events_options():
    return ("", 204)


@app.route("/api/events", methods=["POST"])
def create_event():
    missing = _missing_config()
    if missing:
        return jsonify({"ok": False, "error": f"Missing config: {', '.join(missing)}"}), 500

    if not _is_authorized(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    event = {
        "event_time": data.get("event_time"),
        "event_type": data.get("event_type", "unknown"),
        "confidence": data.get("confidence"),
        "device_id": data.get("device_id", "unknown_device"),
        "source": data.get("source", "edge"),
        "metadata": data.get("metadata", {}),
    }

    if not event["event_time"]:
        return jsonify({"ok": False, "error": "event_time is required"}), 400

    url = f"{_supabase_base_url()}/smartfall_events"
    params = {"select": "id,event_time,event_type,confidence,device_id,source,metadata,created_at"}
    resp = requests.post(url, params=params, json=event, headers=_supabase_headers(True), timeout=10)

    if resp.status_code >= 400:
        return jsonify({"ok": False, "error": resp.text}), 500

    rows = resp.json() if resp.text else []
    return jsonify({"ok": True, "event": rows[0] if rows else event})


@app.route("/api/events", methods=["GET"])
def list_events():
    missing = _missing_config()
    if missing:
        return jsonify({"ok": False, "error": f"Missing config: {', '.join(missing)}"}), 500

    try:
        limit = int(request.args.get("limit", "50"))
    except ValueError:
        limit = 50
    limit = max(1, min(limit, 200))

    url = f"{_supabase_base_url()}/smartfall_events"
    params = {
        "select": "id,event_time,event_type,confidence,device_id,source,metadata,created_at",
        "order": "created_at.desc",
        "limit": str(limit),
    }
    resp = requests.get(url, params=params, headers=_supabase_headers(False), timeout=10)

    if resp.status_code >= 400:
        return jsonify({"ok": False, "error": resp.text}), 500

    return jsonify({"ok": True, "events": resp.json()})
