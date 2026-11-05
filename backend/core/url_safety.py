"""Is a URL safe to fetch from the server? (SSRF protection for URLs that come from search results.)

Search results are untrusted: anyone can publish a page that ranks for a query, and a page can answer with a
redirect to an address of its choosing. Without a check, the server would obediently request
http://169.254.169.254/ (cloud credentials), http://localhost:8000/debug/raw-state (this very API) or a printer on
the office network, and the answer could end up in the report.

Two layers work together:

1. `check_public_url` allows only http and https, resolves the host, and refuses it if ANY resolved address is not
   on the public internet (private, loopback, link-local such as the cloud metadata address, reserved, multicast).
   Because it judges the resolved addresses, decimal or hex spellings of an IP are caught too. The caller
   (RetrievalAgent) runs it before every request, redirects included.
2. `public_only_session` returns a requests session that also verifies the address its socket REALLY connected to,
   before a single byte of the request is sent. Layer 1 resolves the name once and the HTTP library resolves it
   again when it connects, so a hostile DNS server could answer a public address the first time and an internal
   one the second ("DNS rebinding"). Judging the connection itself closes that gap, whatever the DNS says.

Known limit: behind an HTTP proxy the proxy resolves the name and makes the connection, so layer 2 only sees the
proxy's own address. On a shared deployment, block private ranges at the network level too.
"""
import ipaddress
import logging
import socket
from typing import Callable, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http", "https")


def is_public_address(address: str) -> bool:
    """True if `address` (an IP literal) is on the public internet. Raises ValueError if it is not an IP address."""
    ip = ipaddress.ip_address(address.split("%")[0])  # drop an IPv6 scope id such as "%en0"
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped  # ::ffff:127.0.0.1 is loopback written in IPv6
    return ip.is_global and not ip.is_multicast


def check_public_url(url: str, resolve: Callable = socket.getaddrinfo) -> Optional[str]:
    """None if the URL may be fetched, otherwise the reason it may not (a short sentence for the log).

    `resolve` is socket.getaddrinfo; tests pass a fake so that no network is needed.
    """
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return "the URL is not valid"

    if parsed.scheme not in _ALLOWED_SCHEMES:
        return f"the scheme '{parsed.scheme}' is not allowed"
    host = parsed.hostname
    if not host:
        return "the URL has no host"

    try:
        infos = resolve(host, port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):  # socket.gaierror is an OSError; UnicodeError is an invalid host name
        return f"the host '{host}' does not resolve"

    addresses = sorted({info[4][0] for info in infos})
    if not addresses:
        return f"the host '{host}' has no address"
    for address in addresses:
        try:
            public = is_public_address(address)
        except ValueError:
            return f"the host '{host}' resolved to something that is not an IP address"
        if not public:
            return f"'{host}' resolves to a non-public address ({address})"
    return None


# --------------------------------------------------------------------------------------
# Layer 2: judge the connection itself (closes DNS rebinding)
# --------------------------------------------------------------------------------------
class UnsafeAddressError(OSError):
    """A connection went to an address that is not on the public internet."""


def _verify_peer(sock) -> None:
    """Closes `sock` and raises if the address it is really connected to is not public."""
    address = "unknown"
    try:
        address = sock.getpeername()[0]
        public = is_public_address(address)
    except (OSError, ValueError, IndexError):
        public = False
    if not public:
        sock.close()
        logger.warning(f'[BLOCKED] A connection went to a non-public address ({address}); it was closed before '
                       f'any data was sent.')
        raise UnsafeAddressError(f"connected to a non-public address ({address})")


class _CheckedHTTPConnection(HTTPConnection):
    def _new_conn(self):
        sock = super()._new_conn()
        _verify_peer(sock)  # before any request bytes (and, for https, before the TLS handshake)
        return sock


class _CheckedHTTPSConnection(HTTPSConnection):
    def _new_conn(self):
        sock = super()._new_conn()
        _verify_peer(sock)
        return sock


class _CheckedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _CheckedHTTPConnection


class _CheckedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _CheckedHTTPSConnection


class PublicOnlyAdapter(HTTPAdapter):
    """A requests adapter whose connections are verified to be on the public internet."""

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _CheckedHTTPConnectionPool,
            "https": _CheckedHTTPSConnectionPool,
        }


def public_only_session() -> requests.Session:
    """A requests session that refuses to talk to non-public addresses. Create one per download and close it."""
    session = requests.Session()
    adapter = PublicOnlyAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
