from __future__ import annotations

import re
from difflib import SequenceMatcher
from urllib.parse import urlsplit

from nodelm.artifacts import content_digest


def canonical_repository(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError("repository identity cannot be empty")

    has_explicit_host = False
    scp_match = re.fullmatch(r"git@([^:]+):(.+)", candidate)
    if scp_match:
        host = scp_match.group(1)
        path = scp_match.group(2)
        candidate = f"{host}/{path}"
        has_explicit_host = True
    elif "://" in candidate:
        parsed = urlsplit(candidate)
        if not parsed.hostname:
            raise ValueError(f"repository URL has no host: {value}")
        candidate = f"{parsed.hostname}/{parsed.path.lstrip('/')}"
        has_explicit_host = True

    candidate = candidate.removesuffix(".git").strip("/").lower()
    parts = [part for part in candidate.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"repository identity must contain owner and name: {value}")
    if has_explicit_host and len(parts) < 3:
        raise ValueError(f"repository URL must contain owner and name: {value}")
    if not has_explicit_host and len(parts) == 2:
        parts.insert(0, "github.com")
    return "/".join(parts)


def normalize_fingerprint_text(value: str) -> str:
    """Normalize text once for exact hashes, near matching, and safe length bounds."""

    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in value.split("\n")).strip()


def exact_fingerprint(value: str) -> str:
    return content_digest(normalize_fingerprint_text(value).encode("utf-8"))


def is_near_duplicate(left: str, right: str, *, threshold: float) -> bool:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    normalized_left = normalize_fingerprint_text(left)
    normalized_right = normalize_fingerprint_text(right)
    # SequenceMatcher's tie-breaking can be order-sensitive. Duplicate membership must be a
    # symmetric relation, and the contamination gate conservatively treats either direction's
    # measured match as evidence.
    ratio = max(
        SequenceMatcher(None, normalized_left, normalized_right, autojunk=False).ratio(),
        SequenceMatcher(None, normalized_right, normalized_left, autojunk=False).ratio(),
    )
    return ratio >= threshold
