import hashlib
import os
import ssl
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FirewallResult:
    allowed: bool = True
    status_code: int = 200
    reason: str = ""
    ring: int = 1


class FirewallTransport:
    def __init__(
        self,
        tls_enabled: bool = False,
        certfile: Optional[str] = None,
        keyfile: Optional[str] = None,
        cafile: Optional[str] = None,
        require_client_cert: bool = False,
        min_version: int = ssl.TLSVersion.TLSv1_2,
        ciphers: Optional[str] = None,
        pinned_cert_fingerprints: Optional[list] = None,
        alpn_protocols: Optional[list] = None,
    ):
        self.tls_enabled = tls_enabled
        self.certfile = certfile
        self.keyfile = keyfile
        self.cafile = cafile
        self.require_client_cert = require_client_cert
        self.min_version = min_version
        self.ciphers = ciphers
        self.pinned_fingerprints = pinned_cert_fingerprints or []
        self.alpn_protocols = alpn_protocols or ["h2", "http/1.1"]
        self._ssl_context = None
        if self.tls_enabled:
            self._build_context()

    def _build_context(self) -> ssl.SSLContext:
        if not self.certfile or not os.path.isfile(self.certfile):
            raise FileNotFoundError(f"TLS certfile not found: {self.certfile}")
        if not self.keyfile or not os.path.isfile(self.keyfile):
            raise FileNotFoundError(f"TLS keyfile not found: {self.keyfile}")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = self.min_version
        ctx.load_cert_chain(self.certfile, self.keyfile)
        if self.ciphers:
            ctx.set_ciphers(self.ciphers)
        if self.alpn_protocols:
            ctx.set_alpn_protocols(self.alpn_protocols)
        if self.require_client_cert:
            if not self.cafile or not os.path.isfile(self.cafile):
                raise FileNotFoundError(f"CA file required for mTLS: {self.cafile}")
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.load_verify_locations(self.cafile)
        self._ssl_context = ctx
        return ctx

    @property
    def ssl_context(self) -> Optional[ssl.SSLContext]:
        return self._ssl_context

    @staticmethod
    def _pinned_fingerprint(cert_der: bytes) -> str:
        return hashlib.sha256(cert_der).hexdigest()

    def verify_peer_certificate(self, cert_der: bytes) -> FirewallResult:
        if not self.require_client_cert and not self.pinned_fingerprints:
            return FirewallResult(allowed=True, ring=1)
        if not cert_der:
            return FirewallResult(
                allowed=False, status_code=403, reason="no client certificate", ring=1
            )
        fp = self._pinned_fingerprint(cert_der)
        if self.pinned_fingerprints:
            if fp not in self.pinned_fingerprints:
                return FirewallResult(
                    allowed=False,
                    status_code=403,
                    reason="certificate fingerprint not pinned",
                    ring=1,
                )
        return FirewallResult(allowed=True, ring=1)

    def check_request(self, ssl_object: Optional[ssl.SSLObject] = None) -> FirewallResult:
        if not self.tls_enabled:
            return FirewallResult(allowed=True, ring=1)
        if ssl_object is None:
            return FirewallResult(allowed=False, status_code=426, reason="TLS required", ring=1)
        try:
            cert_der = ssl_object.getpeercert(binary_form=True)
        except Exception:
            cert_der = None
        return self.verify_peer_certificate(cert_der or b"")
