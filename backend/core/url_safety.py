"""Is a URL safe to fetch from the server? (SSRF protection for URLs that come from search results.)

Search results are untrusted: anyone can publish a page that ranks for a query, and a page can answer with a
redirect to an address of its choosing. Without a check, the server would obediently request
http://169.254.169.254/ (cloud credentials), http://localhost:8000/debug/raw-state (this very API) or a printer on
the office network, and the answer could end up in the report.

`check_public_url` allows only http and https, resolves the host, and refuses it if ANY resolved address is not on
the public internet (private, loopback, link-local such as the cloud metadata address, reserved, multicast). Because
it judges the resolved addresses, decimal or hex spellings of an IP are caught too. The caller (RetrievalAgent) runs
the check before every request, redirects included, and caps the download size.

Known limit: the name is resolved once for this check and again by the HTTP library when it connects, so a hostile
DNS server could answer differently the second time ("DNS rebinding"). Closing that gap needs connecting to the
checked address directly, or an egress firewall; on a shared deployment, block private ranges at the network level too.
"""
import ipaddress
import socket
from typing import Callable, Optional
from urllib.parse import urlparse

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
