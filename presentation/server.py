import http.server
import socketserver
import json
import os
import argparse
from pathlib import Path

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
        if self.path == '/api/metrics':
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

        elif self.path == '/api/pipeline':
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

        elif self.path.startswith('/api/frame/'):
            stream_dir = Path(__file__).parent.parent / 'output' / 'stream'
            if self.path.startswith('/api/frame/raw'):
                self._serve_jpeg(stream_dir / 'latest_raw.jpg')
            elif self.path.startswith('/api/frame/restored'):
                self._serve_jpeg(stream_dir / 'latest_restored.jpg')
            return

        elif self.path.startswith('/frames/'):
            # /frames/meta.json
            # /frames/corr/0042.jpg
            # /frames/rest/0042.jpg
            rel = self.path[len('/frames/'):].split('?')[0]
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

        elif self.path == '/api/comparison':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            cmp_file = Path(__file__).parent.parent / 'output' / 'model_comparison.json'
            if cmp_file.exists():
                self.wfile.write(cmp_file.read_bytes())
            else:
                self.send_response(404)
                self.wfile.write(b'{"error":"not ready"}')
            return

        elif self.path.startswith('/videos/'):
            fname = self.path[len('/videos/'):].split('?')[0]
            fpath = Path(__file__).parent.parent / 'output' / fname
            if fpath.exists() and fpath.suffix == '.mp4':
                self._serve_video(fpath)
            else:
                self.send_response(404)
                self.end_headers()
            return

        return super().do_GET()

    def _serve_video(self, path: Path):
        """HTTP Range request destekli video - tarayici stream edebilir."""
        try:
            file_size = path.stat().st_size
            range_header = self.headers.get('Range', None)
            if range_header:
                byte_range = range_header.replace('bytes=', '').strip()
                parts = byte_range.split('-')
                start = int(parts[0]) if parts[0] else 0
                end   = int(parts[1]) if parts[1] else file_size - 1
                end   = min(end, file_size - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header('Content-type', 'video/mp4')
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Content-Length', str(length))
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()
                with open(path, 'rb') as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
            else:
                self.send_response(200)
                self.send_header('Content-type', 'video/mp4')
                self.send_header('Content-Length', str(file_size))
                self.send_header('Accept-Ranges', 'bytes')
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
        print(f"Sunum Dashboard'u basladi! Tarayicidan acin: http://localhost:{port}")
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
