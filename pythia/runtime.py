from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Literal

import httpx

from pythia.config import ModelConfig
from pythia.process import ProcessRecord, get_live_record, start_model_server, stop_model

RuntimeState = Literal["stopped", "starting", "ready", "failed"]

CHAT_PROBE_PAYLOAD = {
    "model": "default_model",
    "messages": [{"role": "user", "content": "ping"}],
    "stream": False,
    "max_tokens": 1,
    "temperature": 0,
}


@dataclass(slots=True)
class RuntimeStatus:
    name: str
    model_id: str
    port: int
    state: RuntimeState
    pid: int | None
    last_error: str | None


@dataclass(slots=True)
class RuntimeEntry:
    model: ModelConfig
    state: RuntimeState = "stopped"
    pid: int | None = None
    last_error: str | None = None
    startup_task: asyncio.Task[None] | None = None
    lock: asyncio.Lock | None = None

    def __post_init__(self) -> None:
        self.lock = asyncio.Lock()


class RuntimeManager:
    def __init__(
        self,
        models: list[ModelConfig],
        *,
        startup_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.25,
        http_timeout_seconds: float = 5.0,
        start_server: Callable[..., ProcessRecord] = start_model_server,
        stop_server: Callable[..., object | None] = stop_model,
        live_record_loader: Callable[..., ProcessRecord | None] = get_live_record,
        async_client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        self._entries = {
            model.name: RuntimeEntry(model=model)
            for model in models
        }
        self._startup_timeout_seconds = startup_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._http_timeout_seconds = http_timeout_seconds
        self._start_server = start_server
        self._stop_server = stop_server
        self._live_record_loader = live_record_loader
        self._async_client_factory = async_client_factory

    def get_model(self, alias: str) -> ModelConfig:
        entry = self._entries.get(alias)
        if entry is None:
            raise KeyError(alias)
        return entry.model

    def get_status(self, alias: str) -> RuntimeStatus:
        entry = self._require_entry(alias)
        live_record = self._live_record_loader(alias)
        if live_record is None and entry.state == "ready":
            entry.state = "stopped"
            entry.pid = None
        elif live_record is not None:
            entry.pid = live_record.pid

        return RuntimeStatus(
            name=entry.model.name,
            model_id=entry.model.model_id,
            port=entry.model.port,
            state=entry.state,
            pid=entry.pid,
            last_error=entry.last_error,
        )

    def list_statuses(self) -> list[RuntimeStatus]:
        return [self.get_status(alias) for alias in sorted(self._entries)]

    async def ensure_ready(self, alias: str) -> None:
        entry = self._require_entry(alias)
        task: asyncio.Task[None] | None = None

        async with entry.lock:
            live_record = await asyncio.to_thread(self._live_record_loader, alias)
            if live_record is not None:
                entry.pid = live_record.pid
                if entry.state == "ready":
                    return

            if entry.startup_task is None:
                entry.state = "starting"
                entry.last_error = None
                entry.startup_task = asyncio.create_task(self._start_or_probe(entry))
            task = entry.startup_task

        assert task is not None
        await task

    def _require_entry(self, alias: str) -> RuntimeEntry:
        entry = self._entries.get(alias)
        if entry is None:
            raise KeyError(alias)
        return entry

    async def _start_or_probe(self, entry: RuntimeEntry) -> None:
        started_here = False
        try:
            live_record = await asyncio.to_thread(self._live_record_loader, entry.model.name)
            if live_record is None:
                live_record = await asyncio.to_thread(self._start_server, entry.model)
                started_here = True

            entry.pid = live_record.pid
            await self._wait_until_ready(entry)
            entry.state = "ready"
            entry.last_error = None
        except Exception as error:
            entry.state = "failed"
            entry.last_error = str(error)
            if started_here:
                await asyncio.to_thread(self._stop_server, entry.model.name)
            entry.pid = None
            raise
        finally:
            async with entry.lock:
                entry.startup_task = None

    async def _wait_until_ready(self, entry: RuntimeEntry) -> None:
        deadline = asyncio.get_running_loop().time() + self._startup_timeout_seconds
        base_url = f"http://127.0.0.1:{entry.model.port}"

        while True:
            live_record = await asyncio.to_thread(self._live_record_loader, entry.model.name)
            if live_record is None:
                raise RuntimeError(
                    f"Model '{entry.model.name}' exited before becoming ready"
                )

            entry.pid = live_record.pid
            if await self._probe_ready(base_url):
                return

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for model '{entry.model.name}' to become ready"
                )

            await asyncio.sleep(self._poll_interval_seconds)

    async def _probe_ready(self, base_url: str) -> bool:
        try:
            async with self._async_client_factory(timeout=self._http_timeout_seconds) as client:
                health_response = await client.get(f"{base_url}/health")
                if health_response.status_code != 200:
                    return False

                probe_response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json=CHAT_PROBE_PAYLOAD,
                )
                probe_response.raise_for_status()
                return self._is_valid_chat_completion(probe_response.json())
        except (httpx.HTTPError, ValueError):
            return False

    def _is_valid_chat_completion(self, payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return False
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return False
        message = first_choice.get("message")
        return isinstance(message, dict)
