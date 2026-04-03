from __future__ import annotations

import os
import signal
import tempfile
import unittest
from unittest import mock

from typer.testing import CliRunner

from pythia import cli
from pythia.process import ProcessRecord, StopResult, api_pid_path, read_api_pid, stop_api_server


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
    def __init__(self, cmdline: list[str]) -> None:
        self._cmdline = cmdline
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
    def __init__(self, _config: object, observed: list[int | None]) -> None:
        self._observed = observed

    def run(self) -> None:
        self._observed.append(read_api_pid())


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
        observed: list[int | None] = []

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
        self.assertEqual(observed, [os.getpid()])
        self.assertFalse(api_pid_path().exists())


if __name__ == "__main__":
    unittest.main()
