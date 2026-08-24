from __future__ import annotations

from structlog.testing import capture_logs

from nodelm.logging import configure_structured_logging, get_logger


def test_structured_logging_preserves_machine_readable_context() -> None:
    configure_structured_logging("INFO")

    with capture_logs() as events:
        get_logger(component="fixture").info("check_complete", status="PASS")

    assert events == [
        {
            "component": "fixture",
            "event": "check_complete",
            "log_level": "info",
            "status": "PASS",
        }
    ]
