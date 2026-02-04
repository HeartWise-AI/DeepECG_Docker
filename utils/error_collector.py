from typing import List, Optional


def collect(
    errors: Optional[List[str]],
    source: str,
    message: str,
    detail: Optional[str] = None,
) -> None:
    """
    Append a single error line to the shared errors list (no-op if errors is None).

    Args:
        errors: Mutable list of error strings, or None to skip.
        source: Short label for where the error came from (e.g. "preprocessing", "analysis").
        message: One-line description of the error.
        detail: Optional extra detail appended after the message.
    """
    if errors is None:
        return
    line = f"[{source}] {message}"
    if detail:
        line += f" — {detail}"
    errors.append(line)
