#!/usr/bin/env python3
"""Upload a dataset file or folder to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path


def upload(local_path: Path, repo_id: str, private: bool, num_workers: int) -> None:
    from huggingface_hub import HfApi

    local_path = local_path.expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(local_path)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    if local_path.is_file():
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
        )
    else:
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local_path),
            num_workers=num_workers,
        )

    print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    upload(
        local_path=args.local_path,
        repo_id=args.repo_id,
        private=not args.public,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
