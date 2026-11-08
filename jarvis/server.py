"""
Jarvis -- localhost front end + the glue between the browser, the agents and the panel.

    .venv/Scripts/python server.py              # http://127.0.0.1:8000
    .venv/Scripts/python server.py --serial     # also drive the Arduino panel
    .venv/Scripts/python server.py --serial --port COM6

Pages:  /        choose the CTF (only steganography for now)
        /stego   upload images / videos / audio (+ the expected flag, blank = none planted)
        /run     three agent boxes: cog while working, output on success, soft error on
                 failure, the five metrics + ADI/SR, and restrict / pause / terminate
Stdlib HTTP only, bound to 127.0.0.1.
"""

import argparse
import json
import os
import re
import threading
import time
import uuid
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
import hardware

WEB = os.path.join(config.HERE, "web")


class App:
    def __init__(self, args):
        self.args = args
        self.brain = None
        self.brain_status = "loading model..."
        self.run = None
        self.panel = None
        self.selected = 1
        self._hw_last = {}
        threading.Thread(target=self._load_brain, daemon=True).start()
        if args.serial:
            self.panel = hardware.Panel(args.port, on_encoder=self._on_encoder,
                                        on_select=self._on_select)
            if self.panel.enabled:
                for i, (name, _, _) in enumerate(config.AGENTS, 1):
                    self.panel.set_name(i, name)
                self.panel.calibrate()              # speedometer sweep on start-up
                threading.Thread(target=self._hw_loop, daemon=True).start()
            else:
                print(f"!! panel: {self.panel.last_error}")

    def _load_brain(self):
        try:
            from steering import Brain
            self.brain = Brain(log=print)
            self.brain_status = "ready" if self.brain.steering_ready else \
                "ready (NO desperation vector: run build_desperation_vec.py)"
        except Exception as e:
            self.brain_status = f"model failed to load: {type(e).__name__}: {e}"
        print("model:", self.brain_status)
        if self.panel and self.panel.enabled:
            self.panel.message("model " + ("ready" if self.brain else "ERROR"))

    # ------------------------------------------------------------ hardware
    def _on_encoder(self, idx, delta):
        if self.run and self.run.phase != "finished":
            self.run.encoder(idx, delta)

    def _on_select(self, idx):
        self.selected = idx

    def _hw_loop(self):
        """Push each agent's ADI and LED state; resend unchanged values every 2 s so a
        panel that reset still recovers."""
        last_full = 0
        while self.panel.enabled:
            run = self.run
            if run:
                now = time.time()
                for ag in run.agents:
                    st = (hardware.STATE_SUCCESS if ag.status == "success" else
                          hardware.STATE_FAILED if ag.status == "failed" else
                          hardware.STATE_RUNNING)
                    adi = run.live_adi(ag)          # measured ADI, floored by the time ramp
                    key = (int(round(adi)), st, round(ag.coef, 2))   # c change -> resend now
                    if self._hw_last.get(ag.idx) != key or now - last_full > 2:
                        self.panel.send_agent(ag.idx, adi, st, ag.coef)
                        self._hw_last[ag.idx] = key
                if now - last_full > 2:
                    last_full = now
            time.sleep(config.HW_PERIOD_S)

    # ------------------------------------------------------------ runs
    def start_run(self, files, flag):
        if self.brain is None:
            raise RuntimeError(self.brain_status)
        if self.run and self.run.phase != "finished":
            raise RuntimeError("a run is already in progress")
        import sandbox
        from runner import Run
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        data_dir = os.path.join(config.RUNS_DIR, run_id, "data")
        os.makedirs(data_dir, exist_ok=True)
        for name, blob in files:
            with open(os.path.join(data_dir, name), "wb") as fh:
                fh.write(blob)
        dc = sandbox.client()                     # raises with a clear message if down
        sandbox.ensure_image(dc)
        self._hw_last = {}
        self.run = Run(run_id, data_dir, flag, self.brain,
                       sandbox_factory=lambda rid, ag, d: sandbox.Sandbox(dc, rid, ag, d),
                       telemetry_factory=lambda log: sandbox.Telemetry(dc, log))
        if self.panel and self.panel.enabled:
            self.panel.message("run started")
        self.run.start()
        return run_id

    def state(self):
        s = self.run.snapshot() if self.run else {"phase": "none", "agents": []}
        s["model"] = self.brain_status
        s["selected"] = self.selected
        s["panel"] = (None if not self.args.serial else
                      ("connected " + str(self.panel.port)) if self.panel.enabled
                      else f"not connected ({self.panel.last_error})")
        s["adi_max"] = config.ADI_MAX
        return s


def safe_name(name):
    base = os.path.basename(name.replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".") or "file"
    return base[:80]


def parse_multipart(ctype, body):
    msg = BytesParser(policy=email_policy).parsebytes(
        b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
    files, fields = [], {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        fname = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if fname:
            files.append((name, fname, payload))
        else:
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, default=str))

        def do_GET(self):
            path = self.path.split("?")[0]
            pages = {"/": "index.html", "/stego": "stego.html", "/run": "run.html",
                     "/style.css": "style.css"}
            if path in pages:
                ctype = "text/css" if path.endswith(".css") else "text/html; charset=utf-8"
                with open(os.path.join(WEB, pages[path]), "rb") as fh:
                    return self._send(200, fh.read(), ctype)
            if path == "/api/state":
                return self._json(200, app.state())
            self._send(404, "not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            if n > config.MAX_UPLOAD_MB * 1024 * 1024:
                return self._json(413, {"error": f"upload larger than {config.MAX_UPLOAD_MB} MB"})
            body = self.rfile.read(n)
            if path == "/api/run":
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in ctype:
                    return self._json(400, {"error": "expected multipart/form-data"})
                fields, parts = parse_multipart(ctype, body)
                files, rejected = [], []
                for _, fname, blob in parts:
                    name = safe_name(fname)
                    if os.path.splitext(name)[1].lower() in config.UPLOAD_EXTS and blob:
                        files.append((name, blob))
                    else:
                        rejected.append(fname)
                if not files:
                    return self._json(400, {"error": "upload at least one image, video or "
                                                     "audio file", "rejected": rejected})
                try:
                    rid = app.start_run(files, fields.get("flag", ""))
                except Exception as e:
                    return self._json(503, {"error": str(e)})
                return self._json(200, {"run_id": rid, "rejected": rejected})
            if path == "/api/intervene":
                try:
                    req = json.loads(body or b"{}")
                except json.JSONDecodeError:
                    return self._json(400, {"error": "bad json"})
                if not app.run:
                    return self._json(409, {"error": "no run"})
                ok = app.run.intervene(str(req.get("agent")), str(req.get("action")))
                return self._json(200 if ok else 409, {"ok": ok})
            self._send(404, "not found", "text/plain")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", action="store_true", help="drive the Arduino panel")
    ap.add_argument("--port", help="serial port, e.g. COM6 (default: auto-detect)")
    a = ap.parse_args()
    app = App(a)
    srv = ThreadingHTTPServer((config.HOST, config.PORT), make_handler(app))
    print(f"Jarvis on http://{config.HOST}:{config.PORT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
