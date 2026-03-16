from pathlib import Path


def _default_output_root() -> Path:
    return (Path.cwd() / "../outputs").resolve()


def allocate_output_paths(experiment_name: str, major_version: str, output_root: Path | None = None):
    root = output_root or _default_output_root()
    weights_dir = root / "weights"
    submissions_dir = root / "submissions"
    weights_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{experiment_name}_{major_version}"
    minor = 0
    while True:
        version = f"{major_version}.{minor}"
        stem = f"{prefix}.{minor}"
        weight_path = weights_dir / f"{stem}.pt"
        submission_path = submissions_dir / f"{stem}.csv"
        if not weight_path.exists() and not submission_path.exists():
            return {
                "version": version,
                "weight_path": weight_path,
                "submission_path": submission_path,
            }
        minor += 1
