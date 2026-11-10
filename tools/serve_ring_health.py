"""Local MVP endpoint. Run from repo root with the ring-health environment."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.ring_health.predict import predict


class Handler(BaseHTTPRequestHandler):
    def send_json(self,value,status=200):
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path=="/health":
            self.send_json({"status":"ok","version":"ring-health-v1","mode":"synthetic_demo"})
        else:self.send_json({"error":"not_found"},404)

    def do_POST(self):
        if self.path!="/v1/analyze":
            return self.send_json({"error":"not_found"},404)
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<=2000000:return self.send_json({"error":"body_size"},413)
            payload=json.loads(self.rfile.read(length))
            self.send_json(predict(payload))
        except (ValueError,TypeError,KeyError) as e:self.send_json({"error":"invalid_input","detail":str(e)},400)

    def log_message(self,*args):
        pass  # Do not log health payloads.


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=8787)
    args=parser.parse_args()
    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    print(f"Ring-health demo listening on http://127.0.0.1:{args.port}",flush=True)
    server.serve_forever()
