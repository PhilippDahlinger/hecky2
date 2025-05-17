import http.server
import socketserver
import os

PORT = 8888
ROOT_DIR = "data/labeled_frames"

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def translate_path(self, path):
        # Safely join ROOT_DIR with the requested path
        # So requests like /sub/image.jpg map to ROOT_DIR/sub/image.jpg
        path = os.path.normpath(path.lstrip("/"))
        return os.path.join(ROOT_DIR, path)

if __name__ == '__main__':
    Handler = CORSRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving {ROOT_DIR} at port {PORT} with CORS")
        httpd.serve_forever()
