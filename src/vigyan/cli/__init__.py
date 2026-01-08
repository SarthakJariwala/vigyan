from __future__ import annotations

from .app import app

from . import ingest  # noqa: F401
from . import query  # noqa: F401


def cli() -> None:
    app()
