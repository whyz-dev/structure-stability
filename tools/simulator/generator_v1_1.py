from __future__ import annotations

try:
    from .src_v1_1.runner import main
except ImportError:
    from src_v1_1.runner import main


if __name__ == "__main__":
    main()
