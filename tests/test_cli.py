from __future__ import annotations

import json
import os
import signal
import tempfile
import unittest
from unittest import mock

import httpx
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


class FakeStreamResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        lines: list[str] | None = None,
        json_data: dict[str, object] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._lines = lines or []
        self._json_data = json_data
        self.text = text

    def __enter__(self) -> FakeStreamResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def iter_lines(self) -> list[str]:
        return self._lines

    def json(self) -> dict[str, object]:
        if self._json_data is None:
            raise json.JSONDecodeError("invalid", "", 0)
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://127.0.0.1:11434/api/pull")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)


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

    def test_pull_streams_success(self) -> None:
        response = FakeStreamResponse(
            lines=[
                json.dumps({"status": "pulling manifest", "model": "alpha"}),
                json.dumps({"status": "downloading", "model": "alpha"}),
                json.dumps({"status": "success", "model": "alpha", "completed": 10, "total": 10}),
            ]
        )

        with mock.patch("pythia.cli.httpx.stream", return_value=response) as stream:
            result = self.runner.invoke(cli.app, ["pull", "alpha"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("pulling manifest: alpha", result.stdout)
        self.assertIn("downloading: alpha", result.stdout)
        self.assertIn("success: alpha", result.stdout)
        self.assertIn("Pulled alpha successfully.", result.stdout)
        stream.assert_called_once_with(
            "POST",
            "http://127.0.0.1:11434/api/pull",
            json={"model": "alpha"},
            timeout=None,
        )

    def test_pull_unknown_model_returns_404(self) -> None:
        response = FakeStreamResponse(
            status_code=404,
            json_data={"detail": "Unknown model 'missing'"},
        )

        with mock.patch("pythia.cli.httpx.stream", return_value=response):
            result = self.runner.invoke(cli.app, ["pull", "missing"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("Unknown model 'missing'", result.stdout)

    def test_pull_shows_server_not_running_when_unreachable(self) -> None:
        request = httpx.Request("POST", "http://127.0.0.1:11434/api/pull")
        error = httpx.ConnectError("Connection refused", request=request)

        with mock.patch("pythia.cli.httpx.stream", side_effect=error):
            result = self.runner.invoke(cli.app, ["pull", "alpha"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("Pythia API server is not running. Start it with 'pythia serve'.", result.stdout)


if __name__ == "__main__":
    unittest.main()
