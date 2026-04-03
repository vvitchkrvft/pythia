from __future__ import annotations

import json
import os
import signal
import tempfile
import unittest
from unittest import mock

from typer.testing import CliRunner

from pythia import cli
from pythia.state import (
    api_pid_path,
    runtime_state_path,
    stop_api_server,
)


class FakeApiProcess:
    def __init__(self, cmdline: list[str], pid: int = 1234) -> None:
        self._cmdline = cmdline
        self.pid = pid
        self.signals: list[int] = []

    def is_running(self) -> bool:
        return True

    def status(self) -> str:
        return "running"

    def cmdline(self) -> list[str]:
        return self._cmdline

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def wait(self, timeout: float) -> None:
        return None

    def kill(self) -> None:
        self.signals.append(signal.SIGKILL)


class FakeUvicornServer:
    def __init__(self, _config: object, observed: list[tuple[int | None, int]]) -> None:
        self._observed = observed

    def run(self) -> None:
        payload = json.loads(api_pid_path().read_text(encoding="utf-8"))
        self._observed.append((payload["pid"], payload["port"]))


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_patch = mock.patch.dict(os.environ, {"PYTHIA_HOME": self.temp_dir.name})
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def test_stop_api_server(self) -> None:
        api_pid_path().write_text('{"pid": 1234, "host": "127.0.0.1", "port": 11434}', encoding="utf-8")
        process = FakeApiProcess(["uv", "run", "pythia", "serve"])

        with mock.patch("pythia.state.psutil.Process", return_value=process):
            stopped = stop_api_server()

        self.assertTrue(stopped)
        self.assertEqual(process.signals, [signal.SIGTERM])

    def test_serve_writes_and_cleans_api_pid(self) -> None:
        observed: list[tuple[int | None, int]] = []

        with (
            mock.patch("pythia.cli.ModelRegistry", return_value=mock.Mock()),
            mock.patch("pythia.cli.create_app", return_value=object()),
            mock.patch("pythia.cli.uvicorn.Config", return_value=object()),
            mock.patch(
                "pythia.cli.uvicorn.Server",
                side_effect=lambda config: FakeUvicornServer(config, observed),
            ),
        ):
            result = self.runner.invoke(cli.app, ["serve"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(observed, [(os.getpid(), 11434)])
        self.assertFalse(api_pid_path().exists())

    def test_ps_shows_api_server_and_loaded_model(self) -> None:
        api_pid_path().write_text('{"pid": 1234, "host": "127.0.0.1", "port": 11434}', encoding="utf-8")
        runtime_state_path().write_text(
            '{"name":"alpha","model_id":"repo/alpha","memory_bytes":2048,"idle_expires_at":9999999999.0}',
            encoding="utf-8",
        )
        process = FakeApiProcess(["uv", "run", "pythia", "serve"])

        with mock.patch("pythia.state.psutil.Process", return_value=process):
            result = self.runner.invoke(cli.app, ["ps"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("API Server", result.stdout)
        self.assertIn("Loaded Model", result.stdout)
        self.assertIn("alpha", result.stdout)
        self.assertIn("repo/alpha", result.stdout)

    def test_ps_shows_no_model_when_api_stopped(self) -> None:
        result = self.runner.invoke(cli.app, ["ps"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("API Server", result.stdout)
        self.assertIn("stopped", result.stdout)
        self.assertIn("No model currently loaded.", result.stdout)
        self.assertNotIn("No tracked processes found in ~/.pythia/pids/", result.stdout)


if __name__ == "__main__":
    unittest.main()
