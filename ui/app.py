"""GOAT — Minimalist Web UI with infrastructure visualization."""

import json
import sys
import os

UI_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(UI_DIR, ".."))

from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from goat.config import default_config
from goat.query.pipeline import QueryPipeline


pipeline = QueryPipeline(default_config)


class GOATHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "":
            self._serve_file("index.html", "text/html")
        elif parsed.path == "/api/query":
            self._handle_query(parsed)
        else:
            self.send_error(404)

    def _serve_file(self, filename, content_type):
        filepath = os.path.join(UI_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404)

    def _handle_query(self, parsed):
        params = parse_qs(parsed.query)
        query_text = params.get("q", [""])[0]
        if not query_text:
            self._json_response({"error": "Missing query parameter 'q'"}, 400)
            return

        try:
            response = pipeline.process(query_text)
            result = {
                "query": response.raw_query,
                "intent": response.intent.intent.value if response.intent else "unknown",
                "tier": response.intent.tier.value if response.intent else "?",
                "confidence": round(response.intent.confidence, 2) if response.intent else 0,
                "tokens": list(response.parsed.tokens) if response.parsed else [],
                "entities": response.parsed.entity_mentions if response.parsed else [],
                "periods": response.parsed.periods if response.parsed else [],
                "metrics": response.parsed.metric_mentions if response.parsed else [],
                "time_ms": response.total_time_ms,
                "stages": {
                    "parse": True,
                    "intent": True,
                    "resolve": bool(response.resolution),
                    "expand": bool(response.plan),
                    "plan": bool(response.plan),
                    "retrieve": bool(response.result and response.result.hits),
                    "graph": bool(response.graph_context),
                    "explain": bool(response.explained),
                },
            }
            self._json_response(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json_response({"error": str(e)}, 500)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"  {args[0]}")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    server = HTTPServer(("127.0.0.1", port), GOATHandler)
    print(f"\n  GOAT UI running at http://localhost:{port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
