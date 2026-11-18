"""Serve a run folder on localhost with HTTP Range support, so the audio players can seek (Python's http.server
can't). Symlinks (the page's links to Vāgdhenu's recordings) are followed.

  nandini listen serve [--run out] [--port 8765]
"""
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not m or os.path.isdir(path) or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        a, b = m.groups()
        start, end = (size - int(b), size - 1) if a == "" else (int(a), int(b) if b else size - 1)
        end = min(end, size - 1)
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def copyfile(self, source, outputfile):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(source, outputfile)
        while n > 0:
            buf = source.read(min(65536, n))
            if not buf:
                break
            outputfile.write(buf)
            n -= len(buf)
        self._remaining = None

    def log_message(self, *args):
        pass



def serve(args):
    root = str(Path(args.run).resolve())
    if not (Path(root) / "listen" / "index.html").exists():
        raise SystemExit(f"no page in {root}/listen: run `nandini listen build` first")
    print(f"serving {root}: open http://localhost:{args.port}/listen/  (Ctrl-C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), partial(RangeHandler, directory=root)).serve_forever()
