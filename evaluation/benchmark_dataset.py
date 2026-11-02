"""
Representative SWE Benchmark Instances for Gemma 4 Developer Agent Evaluation.
Covers real-world bug patterns across FastAPI, Requests, Rich, and HTTPX.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BenchmarkTask:
    instance_id: str
    repo: str
    problem_statement: str
    gold_fault_nodes: List[str]
    repo_files: Dict[str, str]
    gold_patch: str
    metadata: Dict[str, str] = field(default_factory=dict)


def get_curated_benchmark_tasks() -> List[BenchmarkTask]:
    """Returns representative curated benchmark tasks reflecting the competition's tasks.jsonl."""
    tasks = []

    # Task 1: FastAPI - Starlette client resolution in TestClient
    fastapi_files = {
        "fastapi/routing.py": '''"""FastAPI routing and APIRoute dispatch."""
from typing import Callable, Any, Optional
from fastapi.exceptions import HTTPException
from fastapi.utils import generate_unique_id
from fastapi.middleware import BaseMiddleware

class APIRoute:
    def __init__(self, path: str, endpoint: Callable, name: Optional[str] = None):
        self.path = path
        self.endpoint = endpoint
        self.name = name or generate_unique_id(path)

    def get_route_handler(self) -> Callable:
        async def custom_route_handler(request: Any) -> Any:
            # Bug: Does not verify if request.client is None before accessing host
            client_host = request.client.host
            response = await self.endpoint(request)
            return response
        return custom_route_handler

def create_api_router(prefix: str = "") -> APIRoute:
    def dummy_endpoint(req):
        return {"status": "ok"}
    return APIRoute(path=prefix, endpoint=dummy_endpoint)
''',
        "fastapi/testclient.py": '''"""FastAPI TestClient wrapper around Starlette."""
from typing import Any
from fastapi.routing import APIRoute
from fastapi.middleware import BaseMiddleware

class TestClient:
    def __init__(self, app: Any, base_url: str = "http://testserver"):
        self.app = app
        self.base_url = base_url

    def get(self, url: str):
        fake_request = type("MockRequest", (), {"client": None, "url": url})()
        route = APIRoute(url, lambda r: "ok")
        handler = route.get_route_handler()
        return handler(fake_request)
''',
        "fastapi/exceptions.py": '''"""FastAPI custom exceptions and status codes."""
class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = status_code
        self.detail = detail

class RequestValidationError(Exception):
    def __init__(self, errors: list):
        self.errors = errors
''',
        "fastapi/utils.py": '''"""Utility functions for route generation and schema creation."""
import hashlib

def generate_unique_id(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:8]

def parse_dependency_injection(func: callable) -> list:
    return getattr(func, "__annotations__", [])
''',
        "fastapi/middleware.py": '''"""Base middleware abstractions."""
class BaseMiddleware:
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        return await self.app(scope, receive, send)
''',
    }
    tasks.append(
        BenchmarkTask(
            instance_id="fastapi__fastapi-1024",
            repo="tiangolo/fastapi",
            problem_statement="""AttributeError: 'NoneType' object has no attribute 'host' when running TestClient.
When making requests using `TestClient` without setting client IP headers, Starlette's `request.client` evaluates to None.
In `APIRoute.get_route_handler`, accessing `request.client.host` crashes instead of safely defaulting to None or empty host.""",
            gold_fault_nodes=["fastapi.routing.APIRoute.get_route_handler"],
            repo_files=fastapi_files,
            gold_patch="""--- a/fastapi/routing.py
+++ b/fastapi/routing.py
@@ -10,3 +10,3 @@ class APIRoute:
-            client_host = request.client.host
+            client_host = request.client.host if request.client else None
""",
        )
    )

    # Task 2: Requests - SSLError with custom cert tuple verification
    requests_files = {
        "requests/sessions.py": '''"""Requests HTTP Session handling."""
from typing import Any
from requests.adapters import HTTPAdapter
from requests.models import Request, Response
from requests.exceptions import SSLError, RequestException
from requests.utils import check_header_validity

class Session:
    def __init__(self):
        self.adapters = {}
        self.cert = None
        self.headers = {}

    def request(self, method: str, url: str, cert: Any = None):
        merged_cert = cert or self.cert
        adapter = self.get_adapter(url)
        return adapter.send(method, url, cert=merged_cert)

    def get_adapter(self, url: str) -> HTTPAdapter:
        return HTTPAdapter()

    def close(self):
        for a in self.adapters.values():
            a.close()
''',
        "requests/adapters.py": '''"""Transport adapters for Requests."""
from typing import Any
from requests.exceptions import SSLError
from requests.models import Response

class HTTPAdapter:
    def __init__(self):
        self.poolmanager = None

    def send(self, method: str, url: str, cert: Any = None):
        # Bug: fails to validate if cert is a tuple (cert_file, key_file)
        if cert and not isinstance(cert, (str, tuple)):
            raise ValueError("Invalid cert format")
        # Bug: tuple length not checked, causing index error on len > 2
        cert_file = cert[0] if isinstance(cert, tuple) else cert
        key_file = cert[1] if isinstance(cert, tuple) else None
        return f"Connected to {url} with cert {cert_file}"

    def close(self):
        pass
''',
        "requests/models.py": '''"""Core request and response objects."""
class Request:
    def __init__(self, method: str, url: str):
        self.method = method
        self.url = url

class Response:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.content = b""
''',
        "requests/exceptions.py": '''"""Requests exceptions hierarchy."""
class RequestException(IOError):
    pass

class SSLError(RequestException):
    pass

class HTTPError(RequestException):
    pass
''',
        "requests/utils.py": '''"""Helper utilities for networking and URL parsing."""
def check_header_validity(header: tuple) -> bool:
    return len(header) == 2 and isinstance(header[0], str)

def super_len(o: Any) -> int:
    return len(o) if hasattr(o, "__len__") else 0
''',
    }
    tasks.append(
        BenchmarkTask(
            instance_id="psf__requests-3179",
            repo="psf/requests",
            problem_statement="""IndexError and SSLError on invalid client certificate tuple in Session.request.
Passing a tuple with invalid length or malformed cert paths into `Session.request` propagates into `HTTPAdapter.send`.
The adapter crashes during ssl context creation rather than performing early validation.""",
            gold_fault_nodes=["requests.adapters.HTTPAdapter.send"],
            repo_files=requests_files,
            gold_patch="""--- a/requests/adapters.py
+++ b/requests/adapters.py
@@ -7,2 +7,4 @@ class HTTPAdapter:
+        if isinstance(cert, tuple) and len(cert) != 2:
+            raise ValueError("Cert tuple must have length 2 (cert, key)")
""",
        )
    )

    # Task 3: Rich - Console text wrapping with zero-width characters
    rich_files = {
        "rich/console.py": '''"""Rich terminal rendering console."""
from typing import Any, List
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

class Console:
    def __init__(self, width: int = 80):
        self.width = width

    def print(self, text: str):
        segment = Segment(text)
        rendered = segment.render(max_width=self.width)
        return rendered

    def export_text(self) -> str:
        return ""
''',
        "rich/segment.py": '''"""Segment representation of printable text."""
from rich.style import Style

class Segment:
    def __init__(self, text: str, style: Style = None):
        self.text = text
        self.style = style

    def cell_length(self) -> int:
        # Bug: naive len() counts zero-width spaces as width 1
        return len(self.text)

    def render(self, max_width: int = 80) -> str:
        actual_len = self.cell_length()
        if actual_len > max_width:
            return self.text[:max_width]
        return self.text
''',
        "rich/style.py": '''"""Style specifications for terminal text."""
class Style:
    def __init__(self, color: str = "white", bold: bool = False):
        self.color = color
        self.bold = bold
''',
        "rich/text.py": '''"""Text markup and spans."""
from rich.style import Style
class Text:
    def __init__(self, plain: str = ""):
        self.plain = plain
        self.spans = []
''',
    }
    tasks.append(
        BenchmarkTask(
            instance_id="textualize__rich-2048",
            repo="textualize/rich",
            problem_statement="""Zero-width unicode characters cause incorrect line wrapping in Console.print.
`Segment.cell_length` calculates width using standard `len(self.text)`, which treats unicode zero-width joiners (ZWJ)
and control sequences as printable glyphs with width 1, resulting in premature line wraps.""",
            gold_fault_nodes=["rich.segment.Segment.cell_length"],
            repo_files=rich_files,
            gold_patch="""--- a/rich/segment.py
+++ b/rich/segment.py
@@ -5,2 +5,3 @@ class Segment:
-        return len(self.text)
+        # Filter out zero-width characters
+        return sum(1 for c in self.text if ord(c) > 31 and ord(c) != 0x200B)
""",
        )
    )

    # Task 4: HTTPX - Stream timeout handling during redirect loop
    httpx_files = {
        "httpx/client.py": '''"""HTTPX async and sync client implementation."""
from typing import Any
from httpx.dispatch import Dispatcher
from httpx.models import Request, Response
from httpx.exceptions import TimeoutException, TooManyRedirects

class Client:
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.dispatcher = Dispatcher()

    def send_handling_redirects(self, request: Any, max_redirects: int = 20):
        # Bug: timeout counter is reset on every redirect hop instead of maintaining overall deadline
        for _ in range(max_redirects):
            response = self.dispatcher.send(request, timeout=self.timeout)
            if not response.is_redirect:
                return response
        raise TooManyRedirects("Too many redirects")
''',
        "httpx/dispatch.py": '''"""Dispatcher transport engine."""
from typing import Any
from httpx.models import Response

class Dispatcher:
    def send(self, request: Any, timeout: float = 5.0):
        return Response(status_code=200, is_redirect=False)
''',
        "httpx/models.py": '''"""Models for HTTP requests and responses."""
class Request:
    def __init__(self, method: str, url: str):
        self.method = method
        self.url = url

class Response:
    def __init__(self, status_code: int = 200, is_redirect: bool = False):
        self.status_code = status_code
        self.is_redirect = is_redirect
''',
        "httpx/exceptions.py": '''"""HTTPX exception definitions."""
class HTTPXError(Exception):
    pass

class TimeoutException(HTTPXError):
    pass

class TooManyRedirects(HTTPXError):
    pass
''',
    }
    tasks.append(
        BenchmarkTask(
            instance_id="encode__httpx-1450",
            repo="encode/httpx",
            problem_statement="""Redirect loop resets timeout deadline indefinitely in Client.send_handling_redirects.
When following redirects in `Client.send_handling_redirects`, the request timeout parameter is reused verbatim
on each redirect step without decrementing elapsed time, allowing redirect chains to exceed the configured client timeout.""",
            gold_fault_nodes=["httpx.client.Client.send_handling_redirects"],
            repo_files=httpx_files,
            gold_patch="""--- a/httpx/client.py
+++ b/httpx/client.py
@@ -8,2 +8,4 @@ class Client:
+        import time
+        start_time = time.monotonic()
""",
        )
    )

    return tasks
