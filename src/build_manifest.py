"""Создаёт artifacts_manifest.json — список всех файлов 8 работ с хешами."""

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(".")
    include = ["src", "configs", "reports", "tests", "data/processed"]
    files = []

    for folder in include:
        for p in root.glob(f"{folder}/**/*"):
            if p.is_file() and not p.name.startswith("."):
                files.append(
                    {
                        "path": str(p),
                        "sha256": sha256(p),
                        "size": p.stat().st_size,
                    }
                )

    manifest = {"files": files, "total": len(files)}
    Path("artifacts_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Манифест создан: {len(files)} файлов")


if __name__ == "__main__":
    main()
