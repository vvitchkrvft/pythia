from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from pythia.config import ModelConfig
from pythia.server import create_app


class StubRuntimeManager:
    def __init__(self, models: list[ModelConfig]) -> None:
        self._models = models

    async def ensure_ready(self, alias: str) -> None:
        return None

    def list_statuses(self) -> list[object]:
        return []


class ServerConfigCachingTests(unittest.TestCase):
    def test_create_app_loads_config_once_not_per_request(self) -> None:
        load_calls = 0
        models = [ModelConfig(name="alpha", model_id="mlx-community/test-model", port=8080)]

        def fake_load_config(_config_path: Path) -> list[ModelConfig]:
            nonlocal load_calls
            load_calls += 1
            return models

        with (
            mock.patch("pythia.server.load_config", side_effect=fake_load_config),
            mock.patch("pythia.server.RuntimeManager", StubRuntimeManager),
            mock.patch("pythia.server.snapshot_download", side_effect=RuntimeError("cache miss")),
            mock.patch("pythia.server.list_process_statuses", return_value=[]),
        ):
            app = create_app(Path("config.yaml"))
            client = TestClient(app)

            first = client.post("/api/show", json={"name": "alpha"})
            second = client.post("/api/show", json={"name": "alpha"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(load_calls, 1)


if __name__ == "__main__":
    unittest.main()
