from __future__ import annotations

try:
    from .src.runner import main
except ImportError:
    from src.runner import main


if __name__ == "__main__":
    main()
