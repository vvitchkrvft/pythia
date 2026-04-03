from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from pythia.registry import ModelRegistry
from pythia.runtime import RuntimeManager
from pythia.state import read_loaded_model_state


class FakeProcess:
    def memory_info(self):
        class Info:
            rss = 2048

        return Info()


class RuntimeManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_pythia_home = os.environ.get("PYTHIA_HOME")
        os.environ["PYTHIA_HOME"] = self.temp_dir.name
        self.addCleanup(self._restore_env)
        self.config_path = Path(self.temp_dir.name) / "config.yaml"
        self.config_path.write_text(
            "models:\n"
            "  - name: alpha\n"
            "    model_id: repo/alpha\n"
            "  - name: beta\n"
            "    model_id: repo/beta\n",
            encoding="utf-8",
        )
        self.registry = ModelRegistry(self.config_path)

    def _restore_env(self) -> None:
        if self.previous_pythia_home is None:
            os.environ.pop("PYTHIA_HOME", None)
        else:
            os.environ["PYTHIA_HOME"] = self.previous_pythia_home

    async def test_reuses_loaded_model_for_same_alias(self) -> None:
        load_calls: list[str] = []

        def fake_load(model_id: str):
            load_calls.append(model_id)
            return object(), FakeTokenizer()

        manager = RuntimeManager(
            self.registry,
            load_model=fake_load,
            process_factory=lambda: FakeProcess(),
            clear_cache=lambda: None,
        )

        async with manager.session("alpha") as first:
            self.assertTrue(first.loaded_now)
        async with manager.session("alpha") as second:
            self.assertFalse(second.loaded_now)

        self.assertEqual(load_calls, ["repo/alpha"])

    async def test_switching_models_unloads_previous_one(self) -> None:
        load_calls: list[str] = []
        clear_calls = 0

        def fake_load(model_id: str):
            load_calls.append(model_id)
            return object(), FakeTokenizer()

        def fake_clear() -> None:
            nonlocal clear_calls
            clear_calls += 1

        manager = RuntimeManager(
            self.registry,
            load_model=fake_load,
            process_factory=lambda: FakeProcess(),
            clear_cache=fake_clear,
        )

        async with manager.session("alpha"):
            pass
        async with manager.session("beta"):
            pass

        self.assertEqual(load_calls, ["repo/alpha", "repo/beta"])
        self.assertGreaterEqual(clear_calls, 1)

    async def test_idle_timeout_unloads_model(self) -> None:
        manager = RuntimeManager(
            self.registry,
            keep_alive_seconds=0,
            load_model=lambda _model_id: (object(), FakeTokenizer()),
            process_factory=lambda: FakeProcess(),
            clear_cache=lambda: None,
        )

        async with manager.session("alpha"):
            pass

        self.assertIsNotNone(read_loaded_model_state())

        await manager.shutdown()
        self.assertIsNone(read_loaded_model_state())

    async def test_idle_timer_clears_loaded_model_state(self) -> None:
        manager = RuntimeManager(
            self.registry,
            keep_alive_seconds=1,
            load_model=lambda _model_id: (object(), FakeTokenizer()),
            process_factory=lambda: FakeProcess(),
            clear_cache=lambda: None,
        )

        async with manager.session("alpha"):
            pass

        self.assertIsNotNone(read_loaded_model_state())
        await asyncio.sleep(1.1)
        self.assertIsNone(read_loaded_model_state())


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


if __name__ == "__main__":
    unittest.main()
