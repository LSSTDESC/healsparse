# tests/_range_server.py  — run as: python _range_server.py <directory>
import sys
from functools import partial
from http.server import ThreadingHTTPServer
from RangeHTTPServer import RangeRequestHandler


class Handler(RangeRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_response(self, code, message=None):
        super().send_response(code, message)
        self.send_header("Connection", "close")
        self.close_connection = True

    def log_message(self, *args):
        pass  # swap to super().log_message(*args) when you need the request log


def main():
    handler = partial(Handler, directory=sys.argv[1])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    print(srv.server_address[1], flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
