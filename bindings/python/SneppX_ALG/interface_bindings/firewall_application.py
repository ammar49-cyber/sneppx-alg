import re
import urllib.parse
from dataclasses import dataclass
from threading import Semaphore
from typing import Dict, List, Optional, Set


@dataclass
class FirewallResult:
    allowed: bool = True
    status_code: int = 200
    reason: str = ""
    ring: int = 3


_INJECTION_PATTERNS = [
    re.compile(r"(?i)(\bunion\b.*\bselect\b|\bselect\b.*\bfrom\b)"),
    re.compile(r"(?i)(\bdrop\b.*\btable\b|\bdrop\b.*\bdatabase\b|\bdelete\b.*\bfrom\b)"),
    re.compile(r"(?i)(\bexec\b|\bexecute\b|\bxp_cmdshell\b|\bsp_executesql\b)"),
    re.compile(r"(?i)(\b(or|and)\s+\d+\s*=\s*\d+\b)"),
    re.compile(r"(?i)(<script[^>]*>|<\/script>|javascript\s*:)"),
    re.compile(r"(?i)(\b(?:cat|tac|tail|nano|vim|vi|nmap|curl|wget)\b)"),
]

_PATH_BYPASS_PATTERNS = [
    (re.compile(r"//+"), "/"),
    (re.compile(r"/\./"), "/"),
    (re.compile(r"/[^/]+/\.\./"), "/"),
    (re.compile(r"%00"), ""),
    (re.compile(r"\\"), "/"),
]

_TRAVERSAL_PATTERNS = [
    re.compile(r"\.\."),
    re.compile(r"%2e%2e", re.IGNORECASE),
    re.compile(r"%252e%252e", re.IGNORECASE),
]


class ConcurrentLimiter:
    def __init__(self, max_per_ip: int = 10):
        self.max_per_ip = max_per_ip
        self._semaphores: Dict[str, Semaphore] = {}

    def acquire(self, ip: str) -> bool:
        sem = self._semaphores.get(ip)
        if sem is None:
            sem = Semaphore(self.max_per_ip)
            self._semaphores[ip] = sem
        return sem.acquire(blocking=False)

    def release(self, ip: str):
        sem = self._semaphores.get(ip)
        if sem is not None:
            sem.release()


class FirewallApplication:
    def __init__(
        self,
        max_body_size: int = 1_048_576,
        allowed_methods: Optional[Set[str]] = None,
        allowed_content_types: Optional[Set[str]] = None,
        max_concurrent_per_ip: int = 10,
        enable_injection_filter: bool = True,
        **kwargs,
    ):
        self.max_body_size = max_body_size
        self.allowed_methods = allowed_methods or {"GET", "POST"}
        self.allowed_content_types = allowed_content_types or {"application/json"}
        self.max_concurrent_per_ip = max_concurrent_per_ip
        self.enable_injection_filter = enable_injection_filter
        self._concurrent = ConcurrentLimiter(max_concurrent_per_ip)
        self._suspicious_headers = {
            "x-forwarded-for",
            "x-forwarded-host",
            "x-http-method-override",
        }

    @staticmethod
    def _normalize_path(path: str) -> str:
        for pattern, replacement in _PATH_BYPASS_PATTERNS:
            prev = None
            while prev != path:
                prev = path
                path = pattern.sub(replacement, path)
        return path

    def _check_body_size(self, body_length: int) -> Optional[FirewallResult]:
        if body_length > self.max_body_size:
            return FirewallResult(
                allowed=False, status_code=413, reason="request body too large", ring=3
            )
        return None

    def _check_method(self, method: str) -> Optional[FirewallResult]:
        if method.upper() not in self.allowed_methods:
            return FirewallResult(
                allowed=False, status_code=405, reason=f"method {method} not allowed", ring=3
            )
        return None

    def _check_content_type(self, content_type: str) -> Optional[FirewallResult]:
        if not content_type:
            return None
        base_type = content_type.split(";")[0].strip().lower()
        if base_type and base_type not in self.allowed_content_types:
            return FirewallResult(
                allowed=False,
                status_code=415,
                reason=f"content type {base_type} not allowed",
                ring=3,
            )
        return None

    def _check_headers(self, headers: dict) -> Optional[FirewallResult]:
        for hdr in self._suspicious_headers:
            if hdr in headers:
                return FirewallResult(
                    allowed=False,
                    status_code=400,
                    reason=f"suspicious header {hdr}",
                    ring=3,
                )
        host = headers.get("host", "")
        if re.search(r"[#<>\[\]\\]", host):
            return FirewallResult(
                allowed=False, status_code=400, reason="malformed host header", ring=3
            )
        return None

    def _check_path(self, path: str) -> Optional[FirewallResult]:
        for segment in path.split("/"):
            if segment == "..":
                return FirewallResult(
                    allowed=False, status_code=400, reason="path traversal detected", ring=3
                )
        for pattern in _TRAVERSAL_PATTERNS:
            if pattern.search(path):
                return FirewallResult(
                    allowed=False, status_code=400, reason="path traversal detected", ring=3
                )
        return None

    def _check_injection(self, body: Optional[bytes], query: str) -> Optional[FirewallResult]:
        if not self.enable_injection_filter:
            return None
        targets = []
        if body:
            try:
                targets.append(body.decode("utf-8", errors="replace"))
            except UnicodeDecodeError:
                pass
        if query:
            targets.append(urllib.parse.unquote(query))
        for target in targets:
            for pattern in _INJECTION_PATTERNS:
                if pattern.search(target):
                    return FirewallResult(
                        allowed=False,
                        status_code=400,
                        reason="injection pattern detected",
                        ring=3,
                    )
        return None

    def check_request(
        self,
        method: str,
        path: str,
        headers: dict,
        body: Optional[bytes] = None,
        query: str = "",
        client_ip: str = "",
    ) -> FirewallResult:
        result = self._check_method(method)
        if result is not None:
            return result
        result = self._check_path(path)
        if result is not None:
            return result
        result = self._check_headers(headers)
        if result is not None:
            return result
        result = self._check_content_type(headers.get("content-type", ""))
        if result is not None:
            return result
        if body is not None:
            result = self._check_body_size(len(body))
            if result is not None:
                return result
        result = self._check_injection(body, query)
        if result is not None:
            return result
        if client_ip:
            if not self._concurrent.acquire(client_ip):
                return FirewallResult(
                    allowed=False,
                    status_code=429,
                    reason="too many concurrent requests",
                    ring=3,
                )
        return FirewallResult(allowed=True)

    def release_concurrent(self, ip: str):
        self._concurrent.release(ip)
