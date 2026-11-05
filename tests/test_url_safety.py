"""The URL safety check (SSRF) and the way RetrievalAgent downloads a page with it. No outside network is used
(a few tests talk to a throw-away server on 127.0.0.1)."""
import http.server
import socket
import threading

import pytest
import requests

from backend.core import url_safety
from backend.core.url_safety import UnsafeAddressError, check_public_url, is_public_address, public_only_session

PUBLIC = "93.184.216.34"


def resolver(*addresses):
    """A fake socket.getaddrinfo that answers with the given addresses."""
    return lambda host, port, type=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]


# --------------------------------------------------------------------------------------
# check_public_url
# --------------------------------------------------------------------------------------
def test_a_public_http_or_https_url_is_allowed():
    assert check_public_url("https://example.org/paper.pdf", resolve=resolver(PUBLIC)) is None
    assert check_public_url("http://example.org:8080/x", resolve=resolver(PUBLIC)) is None


@pytest.mark.parametrize("address", [
    "127.0.0.1",        # loopback: the server itself
    "10.0.0.5", "172.16.0.1", "192.168.1.10",   # private networks
    "169.254.169.254",  # link-local: the cloud metadata service
    "0.0.0.0",          # "this host"
    "100.64.0.1",       # carrier-grade NAT
    "224.0.0.1",        # multicast
    "::1",              # IPv6 loopback
    "fe80::1",          # IPv6 link-local
    "fc00::1",          # IPv6 private
    "::ffff:127.0.0.1",  # loopback written in IPv6
    "fe80::1%en0",      # IPv6 with a scope id
])
def test_a_host_that_resolves_to_a_non_public_address_is_refused(address):
    reason = check_public_url("https://sneaky.example/x", resolve=resolver(address))
    assert reason and "non-public" in reason


def test_one_internal_address_among_public_ones_is_enough_to_refuse():
    assert check_public_url("https://x.example/", resolve=resolver(PUBLIC, "10.0.0.5")) is not None


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.org/paper.pdf",
    "gopher://example.org/",
    "javascript:alert(1)",
])
def test_only_http_and_https_are_allowed(url):
    assert check_public_url(url, resolve=resolver(PUBLIC)) is not None


@pytest.mark.parametrize("url", ["http:///path", "https://", "not a url", "http://example.org:notaport/"])
def test_malformed_urls_are_refused(url):
    assert check_public_url(url, resolve=resolver(PUBLIC)) is not None


def test_a_host_that_does_not_resolve_is_refused():
    def failing(host, port, type=0):
        raise socket.gaierror("no such host")

    assert "does not resolve" in check_public_url("https://nope.invalid/", resolve=failing)


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://127.0.0.1:8000/debug/raw-state/1",
    "http://[::1]/",
    "http://localhost/",
])
def test_addresses_written_into_the_url_are_refused_with_the_real_resolver(url):
    # Numeric hosts resolve without any network access, so this uses the real socket.getaddrinfo.
    assert check_public_url(url) is not None


def test_is_public_address_rejects_something_that_is_not_an_address():
    with pytest.raises(ValueError):
        is_public_address("example.org")


# --------------------------------------------------------------------------------------
# RetrievalAgent._fetch_content
# --------------------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status=200, headers=None, body=b"", encoding="utf-8"):
        self.status_code = status
        self.headers = headers or {}
        self.encoding = encoding
        self._body = body
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start:start + chunk_size]

    def close(self):
        self.closed = True


def redirect(location, status=302):
    return FakeResponse(status=status, headers={"Location": location})


@pytest.fixture
def fetch(rag_module, monkeypatch):
    """(fetch_content, calls, routes): the agent's network is a script. `routes` maps a URL to its FakeResponse and
    `calls` records every URL the agent actually requested."""
    monkeypatch.setattr(rag_module.time, "sleep", lambda seconds: None)
    calls, routes = [], {}

    def fake_get(url, **kwargs):
        calls.append(url)
        return routes[url]

    class FakeSession:
        """Stands in for the checked requests session: the agent's network is a script."""

        def get(self, url, **kwargs):
            return fake_get(url, **kwargs)

        def close(self):
            pass

    monkeypatch.setattr(rag_module, "public_only_session", lambda: FakeSession())
    monkeypatch.setattr(
        rag_module, "check_public_url",
        lambda url: "internal address" if ("169.254." in url or "127.0.0.1" in url) else None,
    )
    return rag_module.RetrievalAgent()._fetch_content, calls, routes


def test_a_normal_page_is_downloaded(fetch):
    fetch_content, calls, routes = fetch
    routes["https://example.org/a"] = FakeResponse(headers={"Content-Type": "text/html; charset=utf-8"}, body="<html>héllo</html>".encode())

    assert fetch_content("https://example.org/a") == {"type": "html", "data": "<html>héllo</html>"}
    assert calls == ["https://example.org/a"]


def test_a_pdf_is_returned_as_a_stream(fetch):
    fetch_content, _, routes = fetch
    routes["https://example.org/p"] = FakeResponse(headers={"Content-Type": "application/pdf"}, body=b"%PDF-1.7 data")

    result = fetch_content("https://example.org/p")

    assert result["type"] == "pdf" and result["data"].getvalue() == b"%PDF-1.7 data"


def test_a_url_that_is_not_public_is_never_requested(fetch):
    fetch_content, calls, _ = fetch
    assert fetch_content("http://169.254.169.254/latest/meta-data/") is None
    assert calls == []  # refused before any request left the server


def test_a_redirect_to_an_internal_address_is_not_followed(fetch):
    fetch_content, calls, routes = fetch
    routes["https://example.org/a"] = redirect("http://169.254.169.254/latest/meta-data/")

    assert fetch_content("https://example.org/a") is None
    assert calls == ["https://example.org/a"]  # the internal address was checked and never requested


def test_a_safe_redirect_is_followed_including_a_relative_one(fetch):
    fetch_content, calls, routes = fetch
    routes["https://example.org/old"] = redirect("/new", status=301)
    routes["https://example.org/new"] = FakeResponse(headers={"Content-Type": "text/html"}, body=b"<html>ok</html>")

    assert fetch_content("https://example.org/old") == {"type": "html", "data": "<html>ok</html>"}
    assert calls == ["https://example.org/old", "https://example.org/new"]


def test_an_endless_chain_of_redirects_stops(fetch, rag_module):
    fetch_content, calls, routes = fetch
    routes["https://example.org/loop"] = redirect("https://example.org/loop")

    assert fetch_content("https://example.org/loop") is None
    assert len(calls) == rag_module.MAX_REDIRECTS + 1


def test_a_download_over_the_size_limit_is_refused_while_streaming(fetch, rag_module, monkeypatch):
    monkeypatch.setattr(rag_module, "MAX_DOWNLOAD_BYTES", 100)
    fetch_content, _, routes = fetch
    big = FakeResponse(headers={"Content-Type": "application/pdf"}, body=b"x" * 500)  # no Content-Length
    routes["https://example.org/big"] = big

    assert fetch_content("https://example.org/big") is None
    assert big.closed


def test_a_declared_size_over_the_limit_is_refused_without_reading(fetch, rag_module, monkeypatch):
    monkeypatch.setattr(rag_module, "MAX_DOWNLOAD_BYTES", 100)
    fetch_content, _, routes = fetch
    declared = FakeResponse(headers={"Content-Type": "application/pdf", "Content-Length": "100000"}, body=b"x")
    routes["https://example.org/declared"] = declared

    assert fetch_content("https://example.org/declared") is None
    assert declared.closed


def test_a_forbidden_page_is_skipped(fetch):
    fetch_content, _, routes = fetch
    routes["https://example.org/waf"] = FakeResponse(status=403)

    assert fetch_content("https://example.org/waf") is None


# --------------------------------------------------------------------------------------
# Total time limit for one download
# --------------------------------------------------------------------------------------
class DrippingResponse(FakeResponse):
    """Delivers its body a few bytes at a time, like a hostile or very slow server."""

    def iter_content(self, chunk_size=1):
        for _ in range(10):
            yield b"x" * 10


def test_a_chain_of_slow_redirects_is_stopped_when_the_time_budget_is_used_up(fetch, rag_module, monkeypatch):
    fetch_content, calls, routes = fetch
    readings = iter([0, 0, 25, 50, 75, 100] + [200] * 20)  # the clock: start, then once per hop
    monkeypatch.setattr(rag_module, "_now", lambda: next(readings))
    routes["https://example.org/loop"] = redirect("https://example.org/loop")

    assert fetch_content("https://example.org/loop") is None
    assert len(calls) == 3  # the 4th hop would start after the 60 s budget has run out


def test_a_body_that_keeps_dripping_past_the_deadline_is_stopped(fetch, rag_module, monkeypatch):
    fetch_content, _, routes = fetch
    readings = iter([0, 0, 30, 61] + [999] * 20)  # start, first hop, after the 1st chunk, after the 2nd chunk
    monkeypatch.setattr(rag_module, "_now", lambda: next(readings))
    slow = DrippingResponse(headers={"Content-Type": "application/pdf"})
    routes["https://example.org/slow"] = slow

    assert fetch_content("https://example.org/slow") is None
    assert slow.closed


def test_a_slow_body_that_still_finishes_in_time_is_accepted(fetch, rag_module, monkeypatch):
    fetch_content, _, routes = fetch
    readings = iter([0, 0] + [5 * n for n in range(1, 30)])  # ten chunks, the clock reaches 50 s: inside the budget
    monkeypatch.setattr(rag_module, "_now", lambda: next(readings))
    routes["https://example.org/ok"] = DrippingResponse(headers={"Content-Type": "application/pdf"})

    assert fetch_content("https://example.org/ok")["type"] == "pdf"


def test_the_session_is_closed_after_every_download_even_when_it_fails(rag_module, monkeypatch):
    monkeypatch.setattr(rag_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(rag_module, "check_public_url", lambda url: None)
    closed = []

    class Session:
        def get(self, url, **kwargs):
            raise OSError("connection refused")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(rag_module, "public_only_session", lambda: Session())
    assert rag_module.RetrievalAgent()._fetch_content("https://example.org/x") is None
    assert closed == [True]


# --------------------------------------------------------------------------------------
# Layer 2: the connection itself is judged (DNS rebinding)
# --------------------------------------------------------------------------------------
@pytest.fixture
def local_server():
    """A real HTTP server on 127.0.0.1 that records every request it receives: (port, paths_seen)."""
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            body = b"hello"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], seen
    server.shutdown()
    server.server_close()


def rebinding_dns(monkeypatch):
    """After this, every name the HTTP library resolves points at 127.0.0.1."""
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, p, family=0, type=0, proto=0, flags=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", p))],
    )


def test_a_connection_to_loopback_is_cut_before_any_request_is_sent(local_server):
    port, seen = local_server
    with public_only_session() as session:
        with pytest.raises(requests.exceptions.ConnectionError):
            session.get(f"http://127.0.0.1:{port}/secret", timeout=5)
    assert seen == []  # the server never received the request


def test_dns_rebinding_is_stopped(local_server, monkeypatch):
    """The name looks public to the first lookup (the URL check) and points at loopback for the second one
    (the HTTP library's own lookup when it connects): the classic rebinding attack."""
    port, seen = local_server
    url = f"http://rebind.example:{port}/secret"
    assert check_public_url(url, resolve=resolver(PUBLIC)) is None  # the URL check is fooled, as it always could be

    rebinding_dns(monkeypatch)
    with public_only_session() as session:
        with pytest.raises(requests.exceptions.ConnectionError):
            session.get(url, timeout=5)
    assert seen == []


def test_the_retrieval_agent_is_protected_end_to_end_against_rebinding(rag_module, local_server, monkeypatch):
    port, seen = local_server
    monkeypatch.setattr(rag_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(rag_module, "check_public_url", lambda url: None)  # the URL check is fooled
    rebinding_dns(monkeypatch)

    assert rag_module.RetrievalAgent()._fetch_content(f"http://rebind.example:{port}/secret") is None
    assert seen == []


def test_a_public_peer_is_allowed_through(local_server, monkeypatch):
    port, seen = local_server
    monkeypatch.setattr(url_safety, "is_public_address", lambda address: True)  # pretend loopback is public
    with public_only_session() as session:
        response = session.get(f"http://127.0.0.1:{port}/ok", timeout=5)
    assert response.status_code == 200 and response.text == "hello"
    assert seen == ["/ok"]


def test_the_https_connection_class_judges_its_peer_too(local_server):
    port, _ = local_server
    connection = url_safety._CheckedHTTPSConnection("127.0.0.1", port, timeout=5)
    with pytest.raises(UnsafeAddressError):
        connection._new_conn()  # the TCP connection succeeds, the verification then cuts it (no TLS handshake)
