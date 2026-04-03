from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

from pythia.config import ModelConfig


@dataclass(slots=True)
class ProcessRecord:
    name: str
    model_id: str
    port: int
    pid: int


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


def ensure_state_dir() -> Path:
    state_dir = Path(os.environ.get("PYTHIA_HOME", Path.home() / ".pythia")) / "pids"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _record_path(model_name: str) -> Path:
    return ensure_state_dir() / f"{model_name}.json"


def _read_record(path: Path) -> ProcessRecord:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return ProcessRecord(**payload)


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


def start_model_server(model: ModelConfig) -> ProcessRecord:
    existing_record = load_record(model.name)
    if existing_record and _is_running(existing_record.pid):
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
    process = subprocess.Popen(  # noqa: S603
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    record = ProcessRecord(
        name=model.name,
        model_id=model.model_id,
        port=model.port,
        pid=process.pid,
    )
    _write_record(record)
    return record


def load_record(model_name: str) -> ProcessRecord | None:
    path = _record_path(model_name)
    if not path.exists():
        return None
    return _read_record(path)


def delete_record(model_name: str) -> None:
    path = _record_path(model_name)
    if path.exists():
        path.unlink()


def stop_model(model_name: str, timeout_seconds: float = 5.0) -> StopResult | None:
    record = load_record(model_name)
    if record is None:
        return None

    was_running = False
    try:
        process = psutil.Process(record.pid)
        if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
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


def stop_all_models(timeout_seconds: float = 5.0) -> list[StopResult]:
    stopped_records: list[StopResult] = []
    for path in sorted(ensure_state_dir().glob("*.json")):
        record = _read_record(path)
        stopped_record = stop_model(record.name, timeout_seconds=timeout_seconds)
        if stopped_record is not None:
            stopped_records.append(stopped_record)
    return stopped_records


def list_process_statuses() -> list[ProcessStatus]:
    statuses: list[ProcessStatus] = []
    for path in sorted(ensure_state_dir().glob("*.json")):
        record = _read_record(path)
        status = "stopped"
        ram_bytes = 0

        try:
            process = psutil.Process(record.pid)
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                status = "running"
                ram_bytes = process.memory_info().rss
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
