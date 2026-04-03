from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import psutil

from pythia.config import ModelConfig


@dataclass(slots=True)
class ProcessRecord:
    name: str
    model_id: str
    port: int
    pid: int
    create_time: float
    cmdline: list[str]


@dataclass(slots=True)
class ProcessStatus:
    name: str
    port: int
    pid: int
    status: str
    ram_bytes: int


@dataclass(slots=True)
class StopResult:
    record: ProcessRecord
    was_running: bool


@dataclass(slots=True)
class ApiServerStatus:
    host: str
    port: int
    pid: int | None
    status: str


WarningCallback = Callable[[str], None]


def _pythia_home() -> Path:
    return Path(os.environ.get("PYTHIA_HOME", Path.home() / ".pythia"))


def ensure_state_dir() -> Path:
    state_dir = _pythia_home() / "pids"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def ensure_log_dir() -> Path:
    log_dir = _pythia_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def api_pid_path() -> Path:
    state_dir = _pythia_home()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "api.pid"


def _record_path(model_name: str) -> Path:
    return ensure_state_dir() / f"{model_name}.json"


def _log_path(model_name: str) -> Path:
    return ensure_log_dir() / f"{model_name}.log"


def _warn(message: str, callback: WarningCallback | None) -> None:
    if callback is not None:
        callback(message)


def _read_record(path: Path, warn: WarningCallback | None = None) -> ProcessRecord | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return ProcessRecord(**payload)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        _warn(f"Skipping corrupted PID file {path.name}: {error}", warn)
        return None


def _write_record(record: ProcessRecord) -> None:
    path = _record_path(record.name)
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(record), file, indent=2)


def _is_running(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _matches_record(process: psutil.Process, record: ProcessRecord) -> bool:
    try:
        return (
            process.create_time() == record.create_time
            and process.cmdline() == record.cmdline
        )
    except psutil.Error:
        return False


def _get_verified_process(
    record: ProcessRecord, warn: WarningCallback | None = None
) -> psutil.Process | None:
    try:
        process = psutil.Process(record.pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            delete_record(record.name)
            return None
        if not _matches_record(process, record):
            delete_record(record.name)
            _warn(
                f"Removed stale PID record for {record.name}: PID {record.pid} belongs to a different process.",
                warn,
            )
            return None
        return process
    except psutil.NoSuchProcess:
        delete_record(record.name)
        return None
    except psutil.Error:
        return None


def start_model_server(model: ModelConfig, warn: WarningCallback | None = None) -> ProcessRecord:
    existing_record = load_record(model.name, warn=warn)
    if existing_record and _get_verified_process(existing_record, warn=warn):
        raise RuntimeError(
            f"Model '{model.name}' is already running with PID {existing_record.pid}"
        )

    command = [
        sys.executable,
        "-m",
        "mlx_lm.server",
        "--model",
        model.model_id,
        "--port",
        str(model.port),
    ]
    log_path = _log_path(model.name)

    try:
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdout=subprocess.DEVNULL,
                stderr=log_file,
                start_new_session=True,
            )
    except OSError as error:
        raise RuntimeError(f"Failed to start model '{model.name}': {error}") from error

    return_code = process.poll()
    if return_code is not None:
        error_output = ""
        try:
            error_output = log_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        details = f" Stderr:\n{error_output}" if error_output else ""
        raise RuntimeError(
            f"Model '{model.name}' exited during startup with code {return_code}.{details}"
        )

    ps_process = psutil.Process(process.pid)
    record = ProcessRecord(
        name=model.name,
        model_id=model.model_id,
        port=model.port,
        pid=process.pid,
        create_time=ps_process.create_time(),
        cmdline=ps_process.cmdline(),
    )
    _write_record(record)
    return record


def load_record(
    model_name: str, warn: WarningCallback | None = None
) -> ProcessRecord | None:
    path = _record_path(model_name)
    if not path.exists():
        return None
    record = _read_record(path, warn=warn)
    if record is None:
        return None
    return record


def get_live_record(
    model_name: str, warn: WarningCallback | None = None
) -> ProcessRecord | None:
    record = load_record(model_name, warn=warn)
    if record is None:
        return None
    process = _get_verified_process(record, warn=warn)
    if process is None:
        return None
    return record


def delete_record(model_name: str) -> None:
    path = _record_path(model_name)
    if path.exists():
        path.unlink()


def write_api_pid(pid: int, host: str = "127.0.0.1", port: int = 11434) -> None:
    path = api_pid_path()
    payload = {"pid": pid, "host": host, "port": port}
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_api_pid_status(warn: WarningCallback | None = None) -> ApiServerStatus:
    path = api_pid_path()
    default_status = ApiServerStatus(
        host="127.0.0.1",
        port=11434,
        pid=None,
        status="stopped",
    )
    if not path.exists():
        return default_status

    try:
        raw_value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        _warn(f"Removed corrupted API PID file: {error}", warn)
        delete_api_pid()
        return default_status

    try:
        if raw_value.startswith("{"):
            payload = json.loads(raw_value)
            pid = int(payload["pid"])
            host = str(payload.get("host", "127.0.0.1"))
            port = int(payload.get("port", 11434))
        else:
            pid = int(raw_value)
            host = "127.0.0.1"
            port = 11434
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        _warn(f"Removed corrupted API PID file: {error}", warn)
        delete_api_pid()
        return default_status

    return ApiServerStatus(host=host, port=port, pid=pid, status="running")


def read_api_pid(warn: WarningCallback | None = None) -> int | None:
    return read_api_pid_status(warn=warn).pid


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


def get_api_server_process(warn: WarningCallback | None = None) -> psutil.Process | None:
    api_status = read_api_pid_status(warn=warn)
    if api_status.pid is None:
        return None

    try:
        process = psutil.Process(api_status.pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            delete_api_pid()
            return None
        if not _looks_like_pythia_api_server(process):
            _warn(
                f"Removed stale API PID file: PID {api_status.pid} is not a Pythia API server.",
                warn,
            )
            delete_api_pid()
            return None
        return process
    except psutil.NoSuchProcess:
        delete_api_pid()
        return None
    except psutil.Error:
        return None


def get_api_server_status(warn: WarningCallback | None = None) -> ApiServerStatus:
    api_status = read_api_pid_status(warn=warn)
    if api_status.pid is None:
        return api_status

    process = get_api_server_process(warn=warn)
    if process is None:
        return ApiServerStatus(
            host=api_status.host,
            port=api_status.port,
            pid=None,
            status="stopped",
        )

    return ApiServerStatus(
        host=api_status.host,
        port=api_status.port,
        pid=process.pid,
        status="running",
    )


def stop_api_server(
    timeout_seconds: float = 5.0,
    warn: WarningCallback | None = None,
) -> bool:
    process = get_api_server_process(warn=warn)
    if process is None:
        return False

    try:
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


def stop_model(
    model_name: str,
    timeout_seconds: float = 5.0,
    warn: WarningCallback | None = None,
) -> StopResult | None:
    record = load_record(model_name, warn=warn)
    if record is None:
        return None

    was_running = False
    try:
        process = _get_verified_process(record, warn=warn)
        if process is not None:
            was_running = True
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

    delete_record(model_name)
    return StopResult(record=record, was_running=was_running)


def stop_all_models(
    timeout_seconds: float = 5.0,
    warn: WarningCallback | None = None,
) -> list[StopResult]:
    stopped_records: list[StopResult] = []
    for path in sorted(ensure_state_dir().glob("*.json")):
        record = _read_record(path, warn=warn)
        if record is None:
            continue
        stopped_record = stop_model(
            record.name,
            timeout_seconds=timeout_seconds,
            warn=warn,
        )
        if stopped_record is not None:
            stopped_records.append(stopped_record)
    return stopped_records


def list_process_statuses(warn: WarningCallback | None = None) -> list[ProcessStatus]:
    statuses: list[ProcessStatus] = []
    for path in sorted(ensure_state_dir().glob("*.json")):
        record = _read_record(path, warn=warn)
        if record is None:
            continue
        status = "stopped"
        ram_bytes = 0

        try:
            process = _get_verified_process(record, warn=warn)
            if process is not None:
                status = "running"
                ram_bytes = process.memory_info().rss
            elif not _record_path(record.name).exists():
                continue
        except psutil.Error:
            status = "stopped"

        statuses.append(
            ProcessStatus(
                name=record.name,
                port=record.port,
                pid=record.pid,
                status=status,
                ram_bytes=ram_bytes,
            )
        )

    return statuses
