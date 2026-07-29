"""Small project-relative path helpers."""

from __future__ import annotations

from pathlib import Path


def as_path(path: str | Path) -> Path:
    """Convert a path-like value without resolving or touching it."""
    return path if isinstance(path, Path) else Path(path)


def ensure_directory(path: str | Path) -> Path:
    """Create and return a directory."""
    directory = as_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_project_path(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve a relative path against an explicit root or current directory."""
    candidate = as_path(path)
    if candidate.is_absolute():
        return candidate
    root = as_path(project_root) if project_root is not None else Path.cwd()
    return root / candidate
