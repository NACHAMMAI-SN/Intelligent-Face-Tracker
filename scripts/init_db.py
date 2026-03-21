"""Initialize SQLite database schema using project config."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import load_config
from src.db import init_database


def main() -> None:
    """Load config and initialize the configured SQLite database."""
    config = load_config("config.json")
    db_path = config["db_path"]
    init_database(db_path)
    print(f"Database initialized successfully at: {db_path}")


if __name__ == "__main__":
    main()
