"""Offline contract tests for the optional creative-discipline writer rule.

Run with::

    PYTHONPATH=. python scripts/test_creative_discipline.py
"""

from __future__ import annotations

import importlib
import os


_FLAG = "CREATIVE_DISCIPLINE"
_SECRET_SENTINEL = "creative-discipline-test-secret"


def _load_ensemble(flag: str | None):
    if flag is None:
        os.environ.pop(_FLAG, None)
    else:
        os.environ[_FLAG] = flag
    os.environ["OPENROUTER_API_KEY"] = _SECRET_SENTINEL

    from app import ensemble

    return importlib.reload(ensemble)


def _legacy_writer_system(ensemble) -> str:
    """Reproduce the pre-ablation writer-system expression byte for byte."""
    return (
        ensemble.WRITE_SYSTEM
        + (ensemble._CONCISE_RULE if ensemble.CONCISE else "")
        + ((" " + ensemble.WRITER_LENGTH_HINT) if ensemble.WRITER_LENGTH_HINT else "")
        + (ensemble._GROUNDING_RULE if ensemble.STRICT_GROUNDING else "")
        + (ensemble._EXEMPLAR_BLOCK if ensemble.EXEMPLARS else "")
    )


class _RejectLogging:
    def __getattr__(self, name: str):
        raise AssertionError(f"writer-system construction unexpectedly accessed log.{name}")


def test_default_and_explicit_off_are_byte_identical_to_legacy_system() -> str:
    default = _load_ensemble(None)
    assert default.CREATIVE_DISCIPLINE is False
    default_system = default._writer_system()
    assert default_system == _legacy_writer_system(default)

    explicit_off = _load_ensemble("0")
    assert explicit_off.CREATIVE_DISCIPLINE is False
    explicit_off_system = explicit_off._writer_system()
    assert explicit_off_system.encode("utf-8") == default_system.encode("utf-8")
    assert explicit_off_system == _legacy_writer_system(explicit_off)
    return explicit_off_system


def test_on_appends_only_the_targeted_creative_rules(off_system: str) -> None:
    enabled = _load_ensemble("1")
    assert enabled.CREATIVE_DISCIPLINE is True

    enabled_system = enabled._writer_system()
    delta = enabled._CREATIVE_DISCIPLINE_RULE
    assert enabled_system == off_system + delta

    lowered = delta.casefold()
    required_rules = (
        "creative humor is framing only",
        "preserve literal scene claims",
        "unseen profession, intent, backstory, future action, or off-screen event",
        "explicit similes or metaphors",
        "never turn a person into a developer",
        "never turn typing into commits or code",
        "at most 2 metaphor or punchline devices per caption",
        'never open with "behold" or "ah yes"',
        "avoid repeating api, endpoint, deployment, or zero-latency patterns",
        "preserve rich factual coverage and length",
        "do not shorten formal or globally cap captions",
    )
    for rule in required_rules:
        assert rule in lowered, f"missing creative-discipline rule: {rule!r}"


def test_flag_does_not_touch_secrets_or_logging() -> None:
    enabled = _load_ensemble("1")
    secret_before = enabled.OR_KEY
    logger_sentinel = _RejectLogging()
    enabled.log = logger_sentinel

    enabled._writer_system()

    assert enabled.OR_KEY == secret_before == _SECRET_SENTINEL
    assert enabled.log is logger_sentinel


def main() -> None:
    original_flag = os.environ.get(_FLAG)
    had_flag = _FLAG in os.environ
    original_secret = os.environ.get("OPENROUTER_API_KEY")
    had_secret = "OPENROUTER_API_KEY" in os.environ
    try:
        off_system = test_default_and_explicit_off_are_byte_identical_to_legacy_system()
        test_on_appends_only_the_targeted_creative_rules(off_system)
        test_flag_does_not_touch_secrets_or_logging()
    finally:
        if had_flag:
            assert original_flag is not None
            os.environ[_FLAG] = original_flag
        else:
            os.environ.pop(_FLAG, None)
        if had_secret:
            assert original_secret is not None
            os.environ["OPENROUTER_API_KEY"] = original_secret
        else:
            os.environ.pop("OPENROUTER_API_KEY", None)

    print("creative_discipline_ok")


if __name__ == "__main__":
    main()
