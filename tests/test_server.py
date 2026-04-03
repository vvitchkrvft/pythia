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
        self._models = [
            ModelConfig(name="alpha", model_id="repo/alpha"),
            ModelConfig(name="qwen3-coder", model_id="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"),
        ]

    def all(self) -> list[ModelConfig]:
        return list(self._models)

    def get(self, name: str) -> ModelConfig | None:
        for model in self._models:
            if model.name == name:
                return model
        return next((model for model in self._models if model.model_id == name), None)


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

    def test_v1_models_returns_openai_model_list(self) -> None:
        client = self.create_client()
        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "list")
        self.assertEqual(response.json()["data"][0]["object"], "model")
        self.assertEqual(response.json()["data"][0]["owned_by"], "pythia")

    def test_v1_model_lookup_supports_model_id(self) -> None:
        client = self.create_client()
        response = client.get("/v1/models/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "qwen3-coder")
        self.assertEqual(response.json()["object"], "model")

    def test_v1_chat_completions_returns_non_streaming_openai_response(self) -> None:
        with mock.patch("pythia.server.generate", return_value="Hello!"):
            client = self.create_client()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3-coder",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], "qwen3-coder")
        self.assertEqual(payload["choices"][0]["message"]["content"], "Hello!")
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    def test_v1_chat_completions_streams_sse_chunks(self) -> None:
        chunks = [
            mock.Mock(text="Hello", finish_reason=None),
            mock.Mock(text="!", finish_reason="stop"),
        ]

        with mock.patch("pythia.server.stream_generate", return_value=iter(chunks)):
            client = self.create_client()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3-coder",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        events = [event for event in response.text.strip().split("\n\n") if event]
        first = json.loads(events[0].removeprefix("data: "))
        second = json.loads(events[1].removeprefix("data: "))
        self.assertEqual(first["object"], "chat.completion.chunk")
        self.assertEqual(first["choices"][0]["delta"]["content"], "Hello")
        self.assertIsNone(first["choices"][0]["finish_reason"])
        self.assertEqual(second["choices"][0]["delta"]["content"], "!")
        self.assertEqual(second["choices"][0]["finish_reason"], "stop")
        self.assertEqual(events[2], "data: [DONE]")

    def test_chat_completions_without_v1_uses_openai_format(self) -> None:
        with mock.patch("pythia.server.generate", return_value="Hello!"):
            client = self.create_client()
            response = client.post(
                "/chat/completions",
                json={
                    "model": "qwen3-coder",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "chat.completion")


if __name__ == "__main__":
    unittest.main()
