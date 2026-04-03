from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import psutil

WarningCallback = Callable[[str], None]


@dataclass(slots=True)
class ApiServerStatus:
    host: str = "127.0.0.1"
    port: int = 11434
    pid: int | None = None
    status: str = "stopped"


@dataclass(slots=True)
class LoadedModelState:
    name: str
    model_id: str
    memory_bytes: int
    idle_expires_at: float | None


def _pythia_home() -> Path:
    return Path(os.environ.get("PYTHIA_HOME", Path.home() / ".pythia"))


def _warn(message: str, callback: WarningCallback | None) -> None:
    if callback is not None:
        callback(message)


def api_pid_path() -> Path:
    state_dir = _pythia_home()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "api.pid"


def runtime_state_path() -> Path:
    state_dir = _pythia_home()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "runtime.json"


def write_api_pid(pid: int, host: str = "127.0.0.1", port: int = 11434) -> None:
    api_pid_path().write_text(
        json.dumps({"pid": pid, "host": host, "port": port}),
        encoding="utf-8",
    )


def delete_api_pid() -> None:
    path = api_pid_path()
    if path.exists():
        path.unlink()


def _looks_like_pythia_api_server(process: psutil.Process) -> bool:
    try:
        cmdline = process.cmdline()
    except psutil.Error:
        return False

    joined = " ".join(cmdline)
    return (
        ("pythia" in joined and "serve" in joined)
        or ("pythia.cli" in joined and "serve" in joined)
    )


def read_api_server_status(warn: WarningCallback | None = None) -> ApiServerStatus:
    path = api_pid_path()
    status = ApiServerStatus()
    if not path.exists():
        return status

    try:
        raw_value = path.read_text(encoding="utf-8").strip()
        if raw_value.startswith("{"):
            payload = json.loads(raw_value)
            pid = int(payload["pid"])
            host = str(payload.get("host", status.host))
            port = int(payload.get("port", status.port))
        else:
            pid = int(raw_value)
            host = status.host
            port = status.port
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        _warn(f"Removed corrupted API PID file: {error}", warn)
        delete_api_pid()
        return status

    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            delete_api_pid()
            return status
        if not _looks_like_pythia_api_server(process):
            _warn(
                f"Removed stale API PID file: PID {pid} is not a Pythia API server.",
                warn,
            )
            delete_api_pid()
            return status
        return ApiServerStatus(host=host, port=port, pid=pid, status="running")
    except psutil.NoSuchProcess:
        delete_api_pid()
        return status
    except psutil.Error:
        return status


def stop_api_server(
    timeout_seconds: float = 5.0,
    warn: WarningCallback | None = None,
) -> bool:
    status = read_api_server_status(warn=warn)
    if status.pid is None:
        return False

    try:
        process = psutil.Process(status.pid)
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=timeout_seconds)
        except psutil.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout_seconds)
    except psutil.NoSuchProcess:
        pass
    except psutil.Error:
        pass

    delete_api_pid()
    return True


def write_loaded_model_state(state: LoadedModelState) -> None:
    runtime_state_path().write_text(json.dumps(asdict(state)), encoding="utf-8")


def read_loaded_model_state(warn: WarningCallback | None = None) -> LoadedModelState | None:
    path = runtime_state_path()
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return LoadedModelState(
            name=str(payload["name"]),
            model_id=str(payload["model_id"]),
            memory_bytes=int(payload["memory_bytes"]),
            idle_expires_at=(
                None
                if payload.get("idle_expires_at") is None
                else float(payload["idle_expires_at"])
            ),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        _warn(f"Removed corrupted runtime state file: {error}", warn)
        delete_loaded_model_state()
        return None


def delete_loaded_model_state() -> None:
    path = runtime_state_path()
    if path.exists():
        path.unlink()


def remaining_idle_seconds(state: LoadedModelState | None) -> float | None:
    if state is None or state.idle_expires_at is None:
        return None
    return max(0.0, state.idle_expires_at - time.time())
