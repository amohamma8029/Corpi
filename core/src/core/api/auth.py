import json
import secrets
from collections.abc import Iterable

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})

_MISSING_OR_INVALID_CREDENTIALS = "Missing or invalid bearer token."
_FORBIDDEN_HOST = "Host header is not allowed."
_FORBIDDEN_ORIGIN = "Origin header is not allowed."


async def _reject(send: Send, *, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    message: Message = {
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
    }
    await send(message)
    await send({"type": "http.response.body", "body": body})


async def _preflight(send: Send, origin: str, request_headers: str) -> None:
    message: Message = {
        "type": "http.response.start",
        "status": 200,
        "headers": [
            (b"access-control-allow-origin", origin.encode("ascii")),
            (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
            (
                b"access-control-allow-headers",
                (request_headers or "authorization").encode("ascii"),
            ),
            (b"vary", b"Origin"),
            (b"content-length", b"0"),
        ],
    }
    await send(message)
    await send({"type": "http.response.body", "body": b""})


def _host_name(host_header: str) -> str:
    host = host_header.strip()
    if host.startswith("["):
        return host.partition("]")[0] + "]"
    return host.rpartition(":")[0] or host


class AuthMiddleware:
    """Reject requests before handlers: local Host, allowed Origin, per-launch bearer token.

    A request without an Origin header is accepted because non-browser clients (pytest,
    curl) never send one; the webview always sends Origin for cross-origin calls.
    The webview's fetches are cross-origin (app origin -> 127.0.0.1), so allowed
    requests get `Access-Control-Allow-Origin` on responses and CORS preflights are
    answered before the token check (preflights carry no Authorization header).
    """

    def __init__(self, app: ASGIApp, token: str, allowed_origins: Iterable[str]) -> None:
        self.app = app
        self.token = token
        self.allowed_origins = frozenset(allowed_origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        origin = headers.get("origin")

        if origin is not None and origin not in self.allowed_origins:
            await _reject(send, status=403, detail=_FORBIDDEN_ORIGIN)
            return

        if scope["method"].upper() == "OPTIONS" and origin is not None:
            await _preflight(send, origin, headers.get("access-control-request-headers", ""))
            return

        if _host_name(headers.get("host", "")) not in LOCAL_HOSTS:
            await _reject(send, status=403, detail=_FORBIDDEN_HOST)
            return

        scheme, _, credentials = headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            credentials.strip(), self.token
        ):
            await _reject(send, status=401, detail=_MISSING_OR_INVALID_CREDENTIALS)
            return

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers.append("Access-Control-Allow-Origin", origin or "*")
                response_headers.append("Vary", "Origin")
            await send(message)

        await self.app(scope, receive, send_with_cors)
