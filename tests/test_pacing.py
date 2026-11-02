"""Request pacing: with LLM_REQUESTS_PER_MINUTE set, the pipeline waits between Gemini calls instead of
provoking 429 errors. The free tier of some models allows 5 requests per minute, and one question makes at
least 6 calls (clean query, intent, planning, query generation, synthesis, evaluation).

No test really waits: the pacer is given a fake clock that only moves when somebody sleeps.
"""
import threading
import types

import pytest
from google.genai import errors as genai_errors

from backend.core import utilities
from backend.core.utilities import RequestPacer


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def pacer_with(per_minute, clock):
    return RequestPacer(per_minute, clock=clock, sleep=clock.sleep)


# --------------------------------------------------------------------------------------
# The pacer itself
# --------------------------------------------------------------------------------------
def test_requests_within_the_limit_go_out_at_once():
    clock = FakeClock()
    pacer = pacer_with(5, clock)
    assert [pacer.wait_for_turn("m") for _ in range(5)] == [0.0] * 5
    assert clock.sleeps == []


def test_the_request_over_the_limit_waits_until_the_first_has_left_the_window():
    clock = FakeClock()
    pacer = pacer_with(5, clock)
    for _ in range(5):
        pacer.wait_for_turn("m")

    assert pacer.wait_for_turn("m") == 61.0  # 60 s window + 1 s margin after the first request (t = 0)
    assert clock.sleeps == [61.0]


def test_the_window_slides_instead_of_resetting_every_minute():
    clock = FakeClock()
    pacer = pacer_with(2, clock)
    pacer.wait_for_turn("m")        # t = 0
    clock.now = 10
    pacer.wait_for_turn("m")        # t = 10
    pacer.wait_for_turn("m")        # must wait for the request at t = 0: until t = 61
    pacer.wait_for_turn("m")        # now t = 61; the request at t = 10 frees a slot at t = 71
    assert clock.sleeps == [51.0, 10.0]


def test_requests_that_are_waiting_reserve_their_turn_one_after_the_other():
    """Callers do not all wake up at once: each reserves a later slot (this is what makes threads safe)."""
    clock = FakeClock()
    pacer = RequestPacer(5, clock=clock, sleep=lambda seconds: None)  # the clock stands still
    delays = [pacer.wait_for_turn("m") for _ in range(12)]
    assert delays == [0.0] * 5 + [61.0] * 5 + [122.0] * 2


def test_each_model_has_its_own_budget():
    clock = FakeClock()
    pacer = pacer_with(1, clock)
    pacer.wait_for_turn("primary")
    assert pacer.wait_for_turn("backup") == 0.0   # a different model has its own quota
    assert pacer.wait_for_turn("primary") == 61.0


@pytest.mark.parametrize("per_minute", [0, -1])
def test_a_limit_of_zero_means_no_pacing(per_minute):
    clock = FakeClock()
    pacer = pacer_with(per_minute, clock)
    assert [pacer.wait_for_turn("m") for _ in range(50)] == [0.0] * 50
    assert clock.sleeps == []


def test_threads_never_exceed_the_limit_together():
    """Eight threads share one pacer with a limit of 3: the reserved send times respect the window."""
    clock = FakeClock()
    pacer = RequestPacer(3, clock=clock, sleep=lambda seconds: None)
    delays, lock = [], threading.Lock()

    def request():
        delay = pacer.wait_for_turn("m")
        with lock:
            delays.append(delay)

    threads = [threading.Thread(target=request) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(delays) == [0.0] * 3 + [61.0] * 3 + [122.0] * 2


# --------------------------------------------------------------------------------------
# Reading the setting
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [("5", 5), (" 12 ", 12), ("", 0), (None, 0), ("abc", 0), ("-3", 0), ("2.5", 0), ("0", 0)],
)
def test_the_limit_is_read_from_the_environment_defensively(raw, expected):
    assert utilities._requests_per_minute(raw) == expected


# --------------------------------------------------------------------------------------
# generate_content_with_retry waits for its turn
# --------------------------------------------------------------------------------------
def fake_client(behaviour):
    asked = []

    def generate_content(*args, **kwargs):
        asked.append(kwargs["model"])
        return behaviour(len(asked), kwargs["model"])

    return types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content)), asked


def install(monkeypatch, per_minute, behaviour, fallbacks=()):
    clock = FakeClock()
    client, asked = fake_client(behaviour)
    monkeypatch.setattr(utilities, "LLM_PACER", pacer_with(per_minute, clock))
    monkeypatch.setattr(utilities, "client", client)
    monkeypatch.setattr(utilities, "LLM_MODEL_FALLBACKS", list(fallbacks))
    monkeypatch.setattr(utilities.time, "sleep", lambda seconds: None)  # the retry layer's own pauses
    return clock, asked


def test_the_sixth_call_of_a_question_waits_for_the_free_tier_window(monkeypatch):
    clock, asked = install(monkeypatch, 5, lambda n, model: "ok")

    for _ in range(6):   # clean query, intent, planning, query generation, synthesis, evaluation
        utilities.generate_content_with_retry(model="primary", contents="x")

    assert len(asked) == 6
    assert clock.sleeps == [61.0]   # only the sixth call had to wait


def test_a_retried_request_counts_against_the_limit_too(monkeypatch):
    def behaviour(n, model):
        if n == 1:
            raise genai_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "busy"}})
        return "ok"

    clock, asked = install(monkeypatch, 2, behaviour)

    utilities.generate_content_with_retry(model="primary", contents="x")   # request 1 fails, request 2 succeeds
    utilities.generate_content_with_retry(model="primary", contents="x")   # request 3 is over the limit

    assert len(asked) == 3
    assert clock.sleeps == [61.0]


def test_a_fallback_model_is_paced_separately_from_the_primary(monkeypatch):
    def behaviour(n, model):
        if model == "primary":
            raise genai_errors.ClientError(429, {"error": {
                "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
                "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}],
            }})
        return "from backup"

    clock, asked = install(monkeypatch, 1, behaviour, fallbacks=["backup"])

    # the single request the primary model is allowed was spent on the failed attempt; the backup has its own budget
    assert utilities.generate_content_with_retry(model="primary", contents="x") == "from backup"
    assert asked == ["primary", "backup"]
    assert clock.sleeps == []


def test_without_a_limit_nothing_is_delayed(monkeypatch):
    clock, asked = install(monkeypatch, 0, lambda n, model: "ok")
    for _ in range(20):
        utilities.generate_content_with_retry(model="primary", contents="x")
    assert clock.sleeps == []
