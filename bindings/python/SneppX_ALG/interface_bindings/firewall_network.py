import ctypes
import ipaddress
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional, Tuple

_HAS_FW_ASM = False
_FW_DLL = None

_lib_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "build", "firewall_core.dll"),
]
for _p in _lib_paths:
    if os.path.isfile(_p):
        try:
            _FW_DLL = ctypes.CDLL(_p)
            _HAS_FW_ASM = True
            break
        except (OSError, ctypes.CError):
            _FW_DLL = None


@dataclass
class CidrRule:
    network: ipaddress.IPv4Network
    allow: bool = True
    log: bool = True


@dataclass
class FirewallResult:
    allowed: bool = True
    status_code: int = 200
    reason: str = ""
    ring: int = 2


class SlidingWindowCounter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = Lock()
        self._buckets: dict = {}

    def allow(self, key: str, now: float = None) -> bool:
        if now is None:
            now = time.time()
        with self._lock:
            ts_list = self._buckets.get(key, [])
            cutoff = now - self.window_seconds
            ts_list = [t for t in ts_list if t > cutoff]
            if len(ts_list) >= self.max_requests:
                self._buckets[key] = ts_list
                return False
            ts_list.append(now)
            self._buckets[key] = ts_list
            return True


class ConnectionTracker:
    def __init__(self, max_connections: int = 1000, timeout: int = 300):
        self.max_connections = max_connections
        self.timeout = timeout
        self._lock = Lock()
        self._table: dict = {}
        self._order: list = []

    def check(self, ip_int: int, now: int = None) -> bool:
        if now is None:
            now = int(time.time())
        with self._lock:
            entry = self._table.get(ip_int)
            if entry is not None:
                ts, count = entry
                if now - ts > self.timeout:
                    self._table[ip_int] = (now, 1)
                    return True
                self._table[ip_int] = (now, count + 1)
                return True
            if len(self._table) >= self.max_connections:
                oldest = self._order.pop(0)
                del self._table[oldest]
            self._table[ip_int] = (now, 1)
            self._order.append(ip_int)
            return True


class FirewallNetwork:
    def __init__(
        self,
        allowlist: Optional[List[str]] = None,
        denylist: Optional[List[str]] = None,
        max_connections: int = 1000,
        rate_limit_max: int = 100,
        rate_window_seconds: float = 60.0,
        block_duration_seconds: int = 300,
        knock_ports: Optional[List[int]] = None,
        knock_window_ms: int = 5000,
        geoip_enabled: bool = False,
        geoip_db_path: Optional[str] = None,
    ):
        self._allowlist = [self._parse_cidr(c) for c in (allowlist or [])]
        self._denylist = [self._parse_cidr(c) for c in (denylist or [])]
        self._allowlist_enabled = bool(allowlist)
        self._denylist_enabled = bool(denylist)
        self._rate_limiter = SlidingWindowCounter(rate_limit_max, rate_window_seconds)
        self._conn_tracker = ConnectionTracker(max_connections, block_duration_seconds)
        self._blocked_ips: dict = {}
        self._block_duration = block_duration_seconds
        self._knock_ports = knock_ports or []
        self._knock_window_ns = knock_window_ms * 1_000_000
        self._knock_state: dict = {}
        self._geoip_enabled = geoip_enabled
        self._geoip_reader = None
        self._asm_ip_match = None
        self._asm_conn_track = None
        self._asm_rate_check = None
        if _HAS_FW_ASM and _FW_DLL is not None:
            try:
                self._asm_ip_match = _FW_DLL.firewall_ip_match
                self._asm_ip_match.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ubyte]
                self._asm_ip_match.restype = ctypes.c_bool
            except AttributeError:
                self._asm_ip_match = None

    @staticmethod
    def _parse_cidr(text: str) -> CidrRule:
        if text.startswith("+"):
            return CidrRule(ipaddress.IPv4Network(text[1:]), allow=True)
        if text.startswith("-"):
            return CidrRule(ipaddress.IPv4Network(text[1:]), allow=False)
        return CidrRule(ipaddress.IPv4Network(text), allow=True)

    @staticmethod
    def _ip_to_int(ip_str: str) -> int:
        return int(ipaddress.IPv4Address(ip_str))

    def _check_allowlist(self, ip_int: int) -> Optional[FirewallResult]:
        if not self._allowlist_enabled:
            return None
        ip_str = str(ipaddress.IPv4Address(ip_int))
        for rule in self._allowlist:
            if ipaddress.IPv4Address(ip_str) in rule.network:
                return None
        return FirewallResult(allowed=False, status_code=403, reason="IP not in allowlist", ring=2)

    def _check_denylist(self, ip_int: int) -> Optional[FirewallResult]:
        if not self._denylist_enabled:
            return None
        ip_str = str(ipaddress.IPv4Address(ip_int))
        for rule in self._denylist:
            if ipaddress.IPv4Address(ip_str) in rule.network:
                return FirewallResult(allowed=False, status_code=403, reason="IP in denylist", ring=2)
        return None

    def _check_blocked(self, ip_int: int, now: float) -> Optional[FirewallResult]:
        expiry = self._blocked_ips.get(ip_int)
        if expiry is not None:
            if now < expiry:
                return FirewallResult(allowed=False, status_code=429, reason="IP temporarily blocked", ring=2)
            del self._blocked_ips[ip_int]
        return None

    def _check_rate_limit(self, ip_str: str) -> Optional[FirewallResult]:
        if not self._rate_limiter.allow(ip_str):
            now = time.time()
            self._blocked_ips[self._ip_to_int(ip_str)] = now + self._block_duration
            return FirewallResult(allowed=False, status_code=429, reason="rate limit exceeded", ring=2)
        return None

    def _check_connection_tracker(self, ip_int: int) -> Optional[FirewallResult]:
        self._conn_tracker.check(ip_int)
        return None

    def _check_port_knock(self, ip_str: str, knock_header: str) -> Optional[FirewallResult]:
        if not self._knock_ports:
            return None
        ports = [int(p.strip()) for p in knock_header.split(",") if p.strip().isdigit()]
        if ports == self._knock_ports:
            self._knock_state[ip_str] = time.time()
            return None
        return FirewallResult(allowed=False, status_code=401, reason="invalid port knock sequence", ring=2)

    def _check_geoip(self, ip_str: str) -> Optional[FirewallResult]:
        if not self._geoip_enabled:
            return None
        return None

    def check_request(self, client_ip: str, now: float = None, knock_header: str = "") -> FirewallResult:
        if now is None:
            now = time.time()
        try:
            ip_int = self._ip_to_int(client_ip)
        except ValueError:
            return FirewallResult(allowed=False, status_code=400, reason="invalid client IP", ring=2)

        result = self._check_blocked(ip_int, now)
        if result is not None:
            return result
        result = self._check_allowlist(ip_int)
        if result is not None:
            return result
        result = self._check_denylist(ip_int)
        if result is not None:
            return result
        if self._knock_ports:
            result = self._check_port_knock(client_ip, knock_header)
            if result is not None:
                return result
        result = self._check_rate_limit(client_ip)
        if result is not None:
            return result
        result = self._check_geoip(client_ip)
        if result is not None:
            return result
        self._check_connection_tracker(ip_int)
        return FirewallResult(allowed=True)

    @property
    def summary(self) -> dict:
        return {
            "allowlist": [str(r.network) for r in self._allowlist],
            "denylist": [str(r.network) for r in self._denylist],
            "blocked_ips": len(self._blocked_ips),
            "rate_limit_max": self._rate_limiter.max_requests,
            "rate_window_seconds": self._rate_limiter.window_seconds,
            "knock_ports": self._knock_ports,
            "has_asm": _HAS_FW_ASM,
        }
