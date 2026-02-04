from typing import List, Optional


def collect(
    errors: Optional[List[str]],
    source: str,
    message: str,
    detail: Optional[str] = None,
) -> None:
    if errors is None:
        return
    line = f"[{source}] {message}"
    if detail:
        line += f" — {detail}"
    errors.append(line)
