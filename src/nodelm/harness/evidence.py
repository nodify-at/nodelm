from __future__ import annotations

import re

_NODE_TEST_COUNT = re.compile(r"^(?:#|\u2139)\s+tests\s+([0-9]+)\s*$", re.MULTILINE)


def parse_node_test_count(output: str) -> int | None:
    """Extract Node's final top-level test count from TAP or spec-reporter output.

    Node 20 emits ``# tests N`` in its TAP summary, while newer releases may emit
    ``tests N`` with an information-symbol prefix. Indented child summaries are deliberately
    ignored so the returned value describes the complete invocation. ``None`` means no
    recognized top-level summary was observed and must not be treated as proof that tests ran.
    """

    matches = _NODE_TEST_COUNT.findall(output)
    return int(matches[-1]) if matches else None
