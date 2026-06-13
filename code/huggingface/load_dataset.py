#!/usr/bin/env python3
"""Download a dataset repo from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path


def download(repo_id: str, local_dir: Path, include: list[str] | None, num_workers: int) -> None:
    from huggingface_hub import snapshot_download

    local_dir = local_dir.expanduser().resolve()
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=include,
        max_workers=num_workers,
    )
    print(f"Downloaded: {local_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", type=Path, default=Path("./hf_dataset"))
    parser.add_argument("--include", action="append")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        include=args.include,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
