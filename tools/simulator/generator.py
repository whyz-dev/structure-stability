from __future__ import annotations

try:
    from .src_v2_0.runner import main
except ImportError:
    from src_v2_0.runner import main


if __name__ == "__main__":
    main()
