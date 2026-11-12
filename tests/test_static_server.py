import sys
import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from static_server import RangeHandler  # noqa: E402


def test_range_requests_are_served(tmp_path):
    (tmp_path / "clip.bin").write_bytes(bytes(range(100)))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/clip.bin"
        req = urllib.request.Request(url, headers={"Range": "bytes=10-19"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206
            assert r.read() == bytes(range(10, 20))
            assert r.headers["Content-Range"] == "bytes 10-19/100"
        with urllib.request.urlopen(url) as r:  # a plain request still gets the whole file
            assert len(r.read()) == 100
    finally:
        srv.shutdown()
