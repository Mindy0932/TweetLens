"""Serve the showcase and local RoBERTa inference. Run with python serve_showcase.py."""
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parent
encoder = None
head = None
lock = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        if self.path != "/api/roberta":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 16000:
                raise ValueError("Request too large or empty")
            origin = self.headers.get("Origin")
            if origin and origin not in ("http://127.0.0.1:8765", "http://localhost:8765"):
                self.send_error(403)
                return
            data = json.loads(self.rfile.read(length))
            text = data.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > 3000:
                raise ValueError("Enter 1–3000 characters")
            checkpoint = ROOT / "artifacts/roberta_frozen/head.joblib"
            if not checkpoint.exists():
                raise ValueError("RoBERTa training has not finished")
            global encoder, head
            with lock:
                if encoder is None:
                    import joblib
                    from roberta_frozen import RobertaEncoder
                    encoder = RobertaEncoder()
                    head = joblib.load(checkpoint)
                probs = head.predict_proba(encoder.encode([text]))[0]
            response = dict(label=str(head.classes_[probs.argmax()]),
                            classes=head.classes_.tolist(), probs=probs.tolist())
            body = json.dumps(response).encode()
            self.send_response(200)
        except (ValueError, json.JSONDecodeError) as error:
            body = json.dumps({"error": str(error)}).encode()
            self.send_response(400)
        except Exception:
            import traceback
            traceback.print_exc()
            body = b'{"error":"Local model unavailable; check server log."}'
            self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("Open http://127.0.0.1:8765/TweetLens_Showcase.html", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
