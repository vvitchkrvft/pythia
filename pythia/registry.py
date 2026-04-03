from __future__ import annotations

from pathlib import Path

from pythia.config import ModelConfig, load_config


class ModelRegistry:
    def __init__(self, config_path: Path = Path("config.yaml")) -> None:
        self._models = load_config(config_path)
        self._models_by_name = {model.name: model for model in self._models}
        self._models_by_id = {model.model_id: model for model in self._models}

    def all(self) -> list[ModelConfig]:
        return list(self._models)

    def get(self, name: str) -> ModelConfig | None:
        return self._models_by_name.get(name) or self._models_by_id.get(name)
