import os
import threading
import hmac
import requests
from flask import Flask, request

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import botcore5 as core

app = Flask(__name__)
SECRET = os.environ.get("WEBHOOK_SECRET", "")
_seen = []


@app.get("/")
def home():
    # open /?setup=YOUR_SECRET once to register the webhook
    s = request.args.get("setup", "")
    if SECRET and s and hmac.compare_digest(s, SECRET):
        url = "https://" + request.headers.get("host", "") + "/hook"
        r = requests.post(f"{core.API}/setWebhook", json={
            "url": url, "secret_token": SECRET,
            "allowed_updates": ["message"], "drop_pending_updates": True}, timeout=20)
        return r.text
    return "ok"


@app.post("/hook")
def hook():
    if not SECRET or not hmac.compare_digest(
            request.headers.get("X-Telegram-Bot-Api-Secret-Token", ""), SECRET):
        return "forbidden", 403
    u = request.get_json(silent=True) or {}
    uid = u.get("update_id")
    if uid in _seen:
        return "ok"
    _seen.append(uid)
    del _seen[:-200]
    try:
        core.handle(u)
    except Exception as e:
        print("hook error", repr(e))
    return "ok"
