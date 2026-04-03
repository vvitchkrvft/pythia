from __future__ import annotations

import os
import signal
import tempfile
import unittest
from unittest import mock

from typer.testing import CliRunner

from pythia import cli
from pythia.process import (
    ProcessRecord,
    StopResult,
    api_pid_path,
    read_api_pid_status,
    stop_api_server,
)


def make_stop_result(name: str, pid: int = 4321, was_running: bool = True) -> StopResult:
    return StopResult(
        record=ProcessRecord(
            name=name,
            model_id=f"mlx-community/{name}",
            port=8080,
            pid=pid,
            create_time=1.0,
            cmdline=["python", "-m", "mlx_lm.server"],
        ),
        was_running=was_running,
    )


class FakeApiProcess:
    def __init__(self, cmdline: list[str], pid: int = 1234) -> None:
        self._cmdline = cmdline
        self.pid = pid
        self.signals: list[int] = []
        self.killed = False

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
        self.killed = True


class FakeUvicornServer:
    def __init__(self, _config: object, observed: list[tuple[int | None, int]]) -> None:
        self._observed = observed

    def run(self) -> None:
        status = read_api_pid_status()
        self._observed.append((status.pid, status.port))


class CliStopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_patch = mock.patch.dict(os.environ, {"PYTHIA_HOME": self.temp_dir.name})
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def test_stop_api_server(self) -> None:
        api_pid_path().write_text("1234\n", encoding="utf-8")
        process = FakeApiProcess(["uv", "run", "pythia", "serve"])

        with mock.patch("pythia.process.psutil.Process", return_value=process):
            stopped = stop_api_server()

        self.assertTrue(stopped)
        self.assertEqual(process.signals, [signal.SIGTERM])
        self.assertFalse(api_pid_path().exists())

    def test_stop_api_server_with_stale_pid_file(self) -> None:
        api_pid_path().write_text("9999\n", encoding="utf-8")
        process = FakeApiProcess(["python", "-m", "http.server"])
        warnings: list[str] = []

        with mock.patch("pythia.process.psutil.Process", return_value=process):
            stopped = stop_api_server(warn=warnings.append)

        self.assertFalse(stopped)
        self.assertFalse(api_pid_path().exists())
        self.assertEqual(process.signals, [])
        self.assertTrue(any("stale API PID file" in warning for warning in warnings))

    def test_stop_all_stops_api_and_models(self) -> None:
        stop_results = [make_stop_result("alpha")]

        with (
            mock.patch("pythia.cli.stop_api_server", return_value=True) as stop_api,
            mock.patch("pythia.cli.stop_all_models", return_value=stop_results) as stop_models,
        ):
            result = self.runner.invoke(cli.app, ["stop", "--all"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Stopped Pythia API server.", result.stdout)
        self.assertIn("Stopped alpha (PID 4321)", result.stdout)
        stop_api.assert_called_once()
        stop_models.assert_called_once()

    def test_serve_writes_and_cleans_api_pid(self) -> None:
        observed: list[tuple[int | None, int]] = []

        with (
            mock.patch("pythia.cli.load_config", return_value=[]),
            mock.patch("pythia.cli.create_app", return_value=object()),
            mock.patch("pythia.cli.uvicorn.Config", return_value=object()),
            mock.patch(
                "pythia.cli.uvicorn.Server",
                side_effect=lambda config: FakeUvicornServer(config, observed),
            ),
            mock.patch("pythia.cli.stop_all_models", return_value=[]),
        ):
            result = self.runner.invoke(cli.app, ["serve"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(observed, [(os.getpid(), 11434)])
        self.assertFalse(api_pid_path().exists())


class CliPsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_patch = mock.patch.dict(os.environ, {"PYTHIA_HOME": self.temp_dir.name})
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def test_ps_shows_running_api_server(self) -> None:
        api_pid_path().write_text('{"pid": 1234, "host": "127.0.0.1", "port": 11435}', encoding="utf-8")
        process = FakeApiProcess(["uv", "run", "pythia", "serve"])

        with (
            mock.patch("pythia.process.psutil.Process", return_value=process),
            mock.patch("pythia.cli.list_process_statuses", return_value=[]),
        ):
            result = self.runner.invoke(cli.app, ["ps"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("API Server", result.stdout)
        self.assertIn("running", result.stdout)
        self.assertIn("1234", result.stdout)
        self.assertIn("127.0.0.1", result.stdout)
        self.assertIn("11435", result.stdout)
        self.assertIn("No tracked model processes", result.stdout)

    def test_ps_cleans_stale_api_pid_file(self) -> None:
        api_pid_path().write_text('{"pid": 9999, "host": "127.0.0.1", "port": 11434}', encoding="utf-8")
        process = FakeApiProcess(["python", "-m", "http.server"])

        with (
            mock.patch("pythia.process.psutil.Process", return_value=process),
            mock.patch("pythia.cli.list_process_statuses", return_value=[]),
        ):
            result = self.runner.invoke(cli.app, ["ps"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("stopped", result.stdout)
        self.assertFalse(api_pid_path().exists())

    def test_ps_cleans_corrupted_api_pid_file(self) -> None:
        api_pid_path().write_text("not-json", encoding="utf-8")

        with mock.patch("pythia.cli.list_process_statuses", return_value=[]):
            result = self.runner.invoke(cli.app, ["ps"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Warning:", result.stdout)
        self.assertIn("stopped", result.stdout)
        self.assertFalse(api_pid_path().exists())

    def test_ps_shows_stopped_api_and_no_models(self) -> None:
        with mock.patch("pythia.cli.list_process_statuses", return_value=[]):
            result = self.runner.invoke(cli.app, ["ps"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("API Server", result.stdout)
        self.assertIn("stopped", result.stdout)
        self.assertIn("127.0.0.1", result.stdout)
        self.assertIn("11434", result.stdout)
        self.assertIn("No tracked model processes", result.stdout)

    def test_ps_shows_api_server_with_model_processes(self) -> None:
        api_pid_path().write_text('{"pid": 1234, "host": "127.0.0.1", "port": 11434}', encoding="utf-8")
        process = FakeApiProcess(["uv", "run", "pythia", "serve"])
        model_statuses = [
            mock.Mock(name="alpha", port=8080, pid=4321, status="running", ram_bytes=1024)
        ]
        model_statuses[0].name = "alpha"
        model_statuses[0].port = 8080
        model_statuses[0].pid = 4321
        model_statuses[0].status = "running"
        model_statuses[0].ram_bytes = 1024

        with (
            mock.patch("pythia.process.psutil.Process", return_value=process),
            mock.patch("pythia.cli.list_process_statuses", return_value=model_statuses),
        ):
            result = self.runner.invoke(cli.app, ["ps"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("API Server", result.stdout)
        self.assertIn("Model Processes", result.stdout)
        self.assertIn("alpha", result.stdout)
        self.assertIn("4321", result.stdout)


if __name__ == "__main__":
    unittest.main()
