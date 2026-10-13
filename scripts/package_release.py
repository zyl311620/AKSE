"""Create a clean ZIP of this repository, excluding private/generated artifacts."""

import argparse
import zipfile
from pathlib import Path


EXCLUDED_DIRECTORIES = {".git", "__pycache__", ".venv", "venv", "outputs", "models",
                        ".cache", ".pytest_cache", "build", "dist"}


def eligible(path, root):
    rel = path.relative_to(root)
    if any(part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info") for part in rel.parts[:-1]):
        return False
    name = path.name
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return False
    return path.suffix not in {".pyc", ".pyo", ".zip", ".log"} and not name.startswith("~$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink() and eligible(p, root))
    # Exclusive creation: never replace an existing archive.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "AKSE/" + path.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("archive integrity verification failed")
    print(f"Packaged {len(files)} files: {output}")


if __name__ == "__main__":
    main()
