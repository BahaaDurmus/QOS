import http.server
import socketserver
import json
import os
import argparse
import cgi
import shutil
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "output" / "uploads"
JOB_FILE = ROOT / "output" / "comparison_job.json"
_process_lock = threading.Lock()
_process_running = False


class QoSDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Sadece hatalari goster, normal GET isteklerini gizle
        if args and str(args[1]) not in ('200', '304'):
            super().log_message(format, *args)

    def end_headers(self):
        if self.path.startswith('/api/'):
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            metrics_file = Path(__file__).parent.parent / 'output' / 'wmn_metrics.json'
            try:
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        data = f.read()
                        if not data.strip():
                            data = "{}"
                else:
                    data = json.dumps({"error": "Metrics not found. Run wmn_simulator.py"})
            except Exception as e:
                data = json.dumps({"error": str(e)})
            self.wfile.write(data.encode('utf-8'))
            return

        elif path == '/api/pipeline':
            # 4 asama verisini birlestir
            base = Path(__file__).parent.parent / 'output'
            result = {}
            for fname, key in [
                ('wmn_metrics.json', 'wmn'),
                ('client_stats.json', 'client'),
            ]:
                p = base / fname
                try:
                    result[key] = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
                except Exception:
                    result[key] = {}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            return

        elif path.startswith('/api/frame/'):
            stream_dir = Path(__file__).parent.parent / 'output' / 'stream'
            if path.startswith('/api/frame/raw'):
                self._serve_jpeg(stream_dir / 'latest_raw.jpg')
            elif path.startswith('/api/frame/restored'):
                self._serve_jpeg(stream_dir / 'latest_restored.jpg')
            return

        elif path.startswith('/frames/'):
            # /frames/meta.json
            # /frames/corr/0042.jpg
            # /frames/rest/0042.jpg
            rel = path[len('/frames/'):]
            fpath = Path(__file__).parent.parent / 'output' / 'frames' / rel
            if fpath.exists():
                ctype = 'application/json' if fpath.suffix == '.json' else 'image/jpeg'
                data = fpath.read_bytes()
                self.send_response(200)
                self.send_header('Content-type', ctype)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'max-age=3600')
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
            return

        elif path == '/api/comparison/status':
            self._json_response(self._read_job())
            return

        elif path == '/api/video-source':
            self._json_response(self._read_video_source())
            return

        elif path == '/api/comparison':
            cmp_file = ROOT / 'output' / 'model_comparison.json'
            if cmp_file.exists():
                self._json_response(json.loads(cmp_file.read_text(encoding='utf-8')))
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error":"not ready"}')
            return

        elif path.startswith('/videos/'):
            fname = path[len('/videos/'):]
            fpath = Path(__file__).parent.parent / 'output' / fname
            if fpath.exists() and fpath.suffix == '.mp4':
                self._serve_video(fpath)
            else:
                self.send_response(404)
                self.end_headers()
            return

        return super().do_GET()

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        if self.path == '/api/comparison/upload':
            self._handle_comparison_upload()
            return
        if self.path == '/api/comparison/default':
            self._handle_comparison_default()
            return
        self.send_error(404)

    def _json_response(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_job(self):
        if JOB_FILE.exists():
            try:
                return json.loads(JOB_FILE.read_text(encoding='utf-8'))
            except Exception:
                pass
        return {"status": "idle", "progress": 0, "message": ""}

    def _read_video_source(self):
        default = ROOT / "input.mp4"
        info = {
            "default_video": "input.mp4",
            "default_exists": default.exists(),
            "default_path": str(default),
            "active": "input.mp4" if default.exists() else None,
            "is_default": True,
            "type": "default",
        }
        src = ROOT / "output" / "video_source.json"
        if src.exists():
            try:
                saved = json.loads(src.read_text(encoding='utf-8'))
                info.update(saved)
            except Exception:
                pass
        frames_meta = ROOT / "output" / "frames" / "meta.json"
        if frames_meta.exists():
            try:
                fm = json.loads(frames_meta.read_text(encoding='utf-8'))
                info["frames_source"] = fm.get("source")
                info["n_frames"] = fm.get("n_frames")
            except Exception:
                pass
        return info

    def _start_comparison_worker(self, video_path: Path, loss: float, max_frames: int, resize: float):
        global _process_running
        with _process_lock:
            if _process_running:
                raise RuntimeError("Baska bir islem devam ediyor")
            _process_running = True

        JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOB_FILE.write_text(
            json.dumps({"status": "processing", "progress": 0, "message": "Kuyruga alindi..."}),
            encoding='utf-8',
        )

        def worker():
            global _process_running
            try:
                from comparison_process import run_job, DEFAULT_MODEL
                run_job(
                    video_path,
                    model_path=DEFAULT_MODEL,
                    loss=loss,
                    max_frames=max_frames,
                    resize=resize,
                )
            except Exception as e:
                JOB_FILE.write_text(
                    json.dumps({"status": "error", "progress": 0, "message": "Hata", "error": str(e)}),
                    encoding='utf-8',
                )
            finally:
                with _process_lock:
                    _process_running = False

        threading.Thread(target=worker, daemon=True).start()

    def _handle_comparison_default(self):
        default = ROOT / "input.mp4"
        if not default.exists():
            self._json_response(
                {"error": "input.mp4 proje kokunde bulunamadi"},
                404,
            )
            return
        loss, max_frames, resize = 0.25, 150, 0.5
        try:
            n = int(self.headers.get('Content-Length', 0))
            if n > 0:
                body = self.rfile.read(n).decode('utf-8')
                if body.strip().startswith('{'):
                    data = json.loads(body)
                    loss = float(data.get('loss', loss))
                    max_frames = int(data.get('max_frames', max_frames))
                    resize = float(data.get('resize', resize))
        except Exception:
            pass
        loss = max(0.05, min(0.5, loss))
        max_frames = max(30, min(600, max_frames))
        resize = max(0.25, min(1.0, resize))
        try:
            self._start_comparison_worker(default, loss, max_frames, resize)
        except RuntimeError as e:
            self._json_response({"error": str(e)}, 409)
            return
        self._json_response({"ok": True, "status": "processing", "video": "input.mp4"})

    def _handle_comparison_upload(self):
        global _process_running
        ctype = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in ctype:
            self._json_response({"error": "multipart/form-data gerekli"}, 400)
            return

        try:
            length = int(self.headers.get('Content-Length', 0))
        except ValueError:
            length = 0
        if length > 200 * 1024 * 1024:
            self._json_response({"error": "Dosya cok buyuk (max 200 MB)"}, 413)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                'REQUEST_METHOD': 'POST',
                'CONTENT_TYPE': ctype,
                'CONTENT_LENGTH': str(length),
            },
        )

        if 'video' not in form:
            self._json_response({"error": "video alani gerekli"}, 400)
            return

        field = form['video']
        if not getattr(field, 'file', None) or not getattr(field, 'filename', None):
            self._json_response({"error": "Gecerli bir video secin"}, 400)
            return

        loss = 0.25
        max_frames = 150
        resize = 0.5
        try:
            if form.getvalue('loss'):
                loss = float(form.getvalue('loss'))
            if form.getvalue('max_frames'):
                max_frames = int(form.getvalue('max_frames'))
            if form.getvalue('resize'):
                resize = float(form.getvalue('resize'))
        except (TypeError, ValueError):
            self._json_response({"error": "Gecersiz parametre"}, 400)
            return

        loss = max(0.05, min(0.5, loss))
        max_frames = max(30, min(600, max_frames))
        resize = max(0.25, min(1.0, resize))

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(field.filename).suffix.lower() or '.mp4'
        if ext not in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
            ext = '.mp4'
        save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        with open(save_path, 'wb') as out:
            shutil.copyfileobj(field.file, out)

        try:
            self._start_comparison_worker(save_path, loss, max_frames, resize)
        except RuntimeError as e:
            self._json_response({"error": str(e)}, 409)
            return
        self._json_response({"ok": True, "status": "processing", "file": save_path.name})

    def _serve_video(self, path: Path):
        """HTTP response ile videoyu indirilebilir olarak gonderir (Range olmadan)."""
        try:
            file_size = path.stat().st_size
            self.send_response(200)
            self.send_header('Content-type', 'video/mp4')
            self.send_header('Content-Disposition', f'attachment; filename="{path.name}"')
            self.send_header('Content-Length', str(file_size))
            self.send_header('Accept-Ranges', 'none')
            self.end_headers()
            with open(path, 'rb') as f:
                self.wfile.write(f.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()

    def _serve_jpeg(self, path: Path):
        if path.exists():
            try:
                data = path.read_bytes()
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(204)
            self.end_headers()

def run_server(port=8080):
    # Change to presentation directory
    os.chdir(Path(__file__).parent)
    
    Handler = QoSDashboardHandler
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Web paneli basladi: http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard sunucusu kapatiliyor...")
            httpd.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help='Dashboard portu')
    args = parser.parse_args()
    run_server(args.port)
