"""Реестр версий моделей: статусы, хеши, переключение."""

import hashlib
import json
from pathlib import Path
from threading import Lock

import joblib

from src.api_config import ModelInfo


class ModelRegistry:
    """Потокобезопасный реестр версий моделей.

    Статусы:
    - ready: модель загружена, но не активна
    - trial: пробная (получает малую долю запросов)
    - working: рабочая (активная)
    - retired: снята
    """

    def __init__(self, registry_path: Path, models_dir: Path):
        self.registry_path = registry_path
        self.models_dir = models_dir
        self._lock = Lock()
        self._versions: dict[str, dict] = {}
        self._active_version: str | None = None
        self._models_cache: dict[str, object] = {}

        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            self._versions = data.get("versions", {})
            self._active_version = data.get("active")
        else:
            self._versions = {}
            self._active_version = None

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(
                {
                    "versions": self._versions,
                    "active": self._active_version,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def register(self, version: str, path: Path, metrics: dict) -> None:
        """Регистрирует новую версию модели."""
        with self._lock:
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            self._versions[version] = {
                "version": version,
                "path": str(path),
                "hash": h,
                "metrics": metrics,
                "status": "ready",
            }
            self._save()

    def promote(self, version: str) -> None:
        """Продвигает версию до working. Старая working → retired."""
        with self._lock:
            if version not in self._versions:
                raise ValueError(f"Версия не найдена: {version}")

            # Снимаем текущую рабочую
            if self._active_version and self._active_version in self._versions:
                self._versions[self._active_version]["status"] = "retired"

            self._versions[version]["status"] = "working"
            self._active_version = version

            # Загружаем модель в кеш (атомарно)
            path = Path(self._versions[version]["path"])
            self._models_cache[version] = joblib.load(path)

            self._save()

    def rollback(self, version: str) -> None:
        """Возвращает указанную версию в working."""
        self.promote(version)

    def get_active(self) -> tuple[str | None, object | None]:
        """Возвращает активную версию и модель."""
        if self._active_version is None:
            return None, None
        if self._active_version not in self._models_cache:
            path = Path(self._versions[self._active_version]["path"])
            self._models_cache[self._active_version] = joblib.load(path)
        return self._active_version, self._models_cache[self._active_version]

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(**v) for v in self._versions.values()]

    @property
    def active(self) -> str | None:
        return self._active_version
