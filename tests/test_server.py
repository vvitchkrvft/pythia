from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from pythia.config import ModelConfig
from pythia.server import create_app


class FakeRegistry:
    def __init__(self, _config_path: Path) -> None:
        self._models = [ModelConfig(name="alpha", model_id="repo/alpha")]

    def all(self) -> list[ModelConfig]:
        return list(self._models)

    def get(self, name: str) -> ModelConfig | None:
        return next((model for model in self._models if model.name == name), None)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "CHAT:" + "|".join(message["content"] for message in messages)


class FakeRuntimeManager:
    def __init__(self, _registry: FakeRegistry) -> None:
        self.current = None

    def needs_load(self, alias: str) -> bool:
        return self.current != alias

    @asynccontextmanager
    async def session(self, alias: str):
        loaded_now = self.current != alias
        self.current = alias
        yield mock.Mock(
            alias=alias,
            model_id=f"repo/{alias}",
            model=object(),
            tokenizer=FakeTokenizer(),
            loaded_now=loaded_now,
        )

    def current_model(self):
        if self.current is None:
            return None
        return mock.Mock(name=self.current)

    async def unload(self, alias: str | None = None) -> bool:
        if alias is None or alias == self.current:
            self.current = None
            return True
        return False

    async def shutdown(self) -> None:
        self.current = None


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_pythia_home = os.environ.get("PYTHIA_HOME")
        os.environ["PYTHIA_HOME"] = self.temp_dir.name
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        if self.previous_pythia_home is None:
            os.environ.pop("PYTHIA_HOME", None)
        else:
            os.environ["PYTHIA_HOME"] = self.previous_pythia_home

    def create_client(self):
        with (
            mock.patch("pythia.server.ModelRegistry", FakeRegistry),
            mock.patch("pythia.server.RuntimeManager", FakeRuntimeManager),
            mock.patch("pythia.server.snapshot_download", side_effect=RuntimeError("cache miss")),
        ):
            app = create_app(Path("config.yaml"))
        return TestClient(app)

    def test_generate_uses_in_process_generation(self) -> None:
        with mock.patch("pythia.server.generate", return_value="hello world"):
            client = self.create_client()
            response = client.post(
                "/api/generate",
                json={"model": "alpha", "prompt": "hi", "stream": False},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "hello world")

    def test_chat_stream_emits_loading_line_then_tokens(self) -> None:
        chunks = [
            mock.Mock(text="hello", finish_reason=None),
            mock.Mock(text=" world", finish_reason="stop"),
        ]

        with mock.patch("pythia.server.stream_generate", return_value=iter(chunks)):
            client = self.create_client()
            response = client.post(
                "/api/chat",
                json={
                    "model": "alpha",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        lines = [json.loads(line) for line in response.text.strip().splitlines()]
        self.assertEqual(lines[0]["status"], "loading model")
        self.assertEqual(lines[1]["message"]["content"], "hello")
        self.assertTrue(lines[2]["done"])

    def test_tags_returns_available_when_no_model_loaded(self) -> None:
        client = self.create_client()
        response = client.get("/api/tags")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["models"][0]["status"], "available")

    def test_tags_returns_loaded_for_current_model_from_runtime_state(self) -> None:
        with mock.patch("pythia.server.read_loaded_model_state") as read_state:
            read_state.return_value = mock.Mock()
            read_state.return_value.name = "alpha"
            client = self.create_client()
            response = client.get("/api/tags")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["models"][0]["status"], "loaded")


if __name__ == "__main__":
    unittest.main()
