_FORBIDDEN_EXACT_LINES = {
    "GIT binary patch",
}
_FORBIDDEN_PREFIXES = (
    "Binary files ",
    "literal ",
    "delta ",
    "new file mode ",
    "deleted file mode ",
    "old mode ",
    "new mode ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
)


def validate_text_git_patch(patch: str, *, max_bytes: int = 1_000_000) -> str:
    """Validate a bounded text-only Git patch before invoking ``git apply``."""

    if not isinstance(patch, str) or not patch.strip():
        raise ValueError("patch must be non-empty text")
    if max_bytes <= 0:
        raise ValueError("patch byte limit must be positive")
    if len(patch.encode("utf-8")) > max_bytes:
        raise ValueError("patch exceeds the configured byte limit")
    if "\0" in patch:
        raise ValueError("patch contains a NUL byte")
    if not patch.startswith("diff --git "):
        raise ValueError("patch must start with a git-style unified diff")
    for line in patch.splitlines():
        if line in _FORBIDDEN_EXACT_LINES or line.startswith(_FORBIDDEN_PREFIXES):
            raise ValueError("binary, file-creation, mode, rename, and copy patches are forbidden")
    return patch
