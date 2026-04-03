from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class ModelConfig:
    name: str
    model_id: str
    port: int


def load_config(config_path: Path) -> list[ModelConfig]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    models = raw_config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("config.yaml must define a non-empty 'models' list")

    parsed_models: list[ModelConfig] = []
    seen_names: set[str] = set()
    seen_ports: set[int] = set()

    for index, raw_model in enumerate(models, start=1):
        if not isinstance(raw_model, dict):
            raise ValueError(f"Model entry #{index} must be a mapping")

        name = raw_model.get("name")
        model_id = raw_model.get("model_id")
        port = raw_model.get("port")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Model entry #{index} is missing a valid 'name'")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"Model '{name}' is missing a valid 'model_id'")
        if not isinstance(port, int):
            raise ValueError(f"Model '{name}' is missing a valid integer 'port'")
        if name in seen_names:
            raise ValueError(f"Duplicate model name in config: {name}")
        if port in seen_ports:
            raise ValueError(f"Duplicate port in config: {port}")

        seen_names.add(name)
        seen_ports.add(port)
        parsed_models.append(ModelConfig(name=name, model_id=model_id, port=port))

    return parsed_models
