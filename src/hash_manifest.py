"""Генерация и верификация манифеста хешей сырых данных."""

import hashlib
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
MANIFEST = Path("data/hash_manifest.json")


def sha256_file(path: Path) -> str:
    """Считает SHA-256 хеш файла."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest() -> dict:
    """Собирает манифест по всем файлам в data/raw/."""
    files = []
    for p in sorted(RAW_DIR.glob("*")):
        if p.name.startswith("."):
            continue
        files.append(
            {
                "path": str(p),
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
        )
    return {"files": files}


def verify_manifest() -> bool:
    """Проверяет, что хеши в манифесте совпадают с реальными."""
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ok = True
    for entry in m["files"]:
        actual = sha256_file(Path(entry["path"]))
        if actual != entry["sha256"]:
            ok = False
            print(f"НЕ СОВПАДАЕТ: {entry['path']}")
        else:
            print(f"OK: {entry['path']}")
    print("ИТОГ: все хеши совпадают." if ok else "ИТОГ: есть расхождения.")
    return ok


if __name__ == "__main__":
    manifest = build_manifest()
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Манифест сохранён: {MANIFEST}")
    verify_manifest()
