from __future__ import annotations

import asyncio
import gc
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import psutil
from mlx_lm import load

from pythia.config import ModelConfig
from pythia.registry import ModelRegistry
from pythia.state import LoadedModelState, delete_loaded_model_state, write_loaded_model_state


@dataclass(slots=True)
class LoadedModelSession:
    alias: str
    model_id: str
    model: object
    tokenizer: object
    loaded_now: bool


class RuntimeManager:
    def __init__(
        self,
        registry: ModelRegistry,
        *,
        keep_alive_seconds: int | None = None,
        load_model: Callable[..., tuple[object, object]] = load,
        process_factory: Callable[..., psutil.Process] = psutil.Process,
        clear_cache: Callable[[], None] = mx.metal.clear_cache,
    ) -> None:
        self._registry = registry
        self._keep_alive_seconds = self._parse_keep_alive(keep_alive_seconds)
        self._load_model = load_model
        self._process_factory = process_factory
        self._clear_cache = clear_cache
        self._slot_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._idle_task: asyncio.Task[None] | None = None
        self._loaded_config: ModelConfig | None = None
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._loaded_at: float | None = None
        self._idle_expires_at: float | None = None

    @staticmethod
    def _parse_keep_alive(explicit_keep_alive: int | None) -> int:
        if explicit_keep_alive is not None:
            return max(0, explicit_keep_alive)
        raw_value = os.environ.get("PYTHIA_KEEP_ALIVE", "300")
        try:
            return max(0, int(raw_value))
        except ValueError as error:
            raise ValueError("PYTHIA_KEEP_ALIVE must be an integer number of seconds") from error

    def needs_load(self, alias: str) -> bool:
        return self._loaded_config is None or self._loaded_config.name != alias

    def current_model(self) -> LoadedModelState | None:
        if self._loaded_config is None:
            return None
        return LoadedModelState(
            name=self._loaded_config.name,
            model_id=self._loaded_config.model_id,
            memory_bytes=self._current_memory_bytes(),
            idle_expires_at=self._idle_expires_at,
        )

    async def ensure_ready(self, alias: str) -> LoadedModelSession:
        config = self._registry.get(alias)
        if config is None:
            raise KeyError(alias)

        async with self._slot_lock:
            self._cancel_idle_unload_locked()
            loaded_now = False
            if self._loaded_config is None or self._loaded_config.name != alias:
                await self._unload_locked()
                model, tokenizer = await asyncio.to_thread(
                    self._load_model,
                    config.model_id,
                )
                self._loaded_config = config
                self._model = model
                self._tokenizer = tokenizer
                self._loaded_at = time.time()
                loaded_now = True

            self._refresh_idle_deadline_locked()
            self._write_runtime_state_locked()

            assert self._model is not None
            assert self._tokenizer is not None
            return LoadedModelSession(
                alias=config.name,
                model_id=config.model_id,
                model=self._model,
                tokenizer=self._tokenizer,
                loaded_now=loaded_now,
            )

    @asynccontextmanager
    async def session(self, alias: str):
        async with self._inference_lock:
            session = await self.ensure_ready(alias)
            try:
                yield session
            finally:
                async with self._slot_lock:
                    self._refresh_idle_deadline_locked()
                    self._write_runtime_state_locked()
                    self._schedule_idle_unload_locked()

    async def unload(self, alias: str | None = None) -> bool:
        async with self._slot_lock:
            if self._loaded_config is None:
                return False
            if alias is not None and self._loaded_config.name != alias:
                return False
            await self._unload_locked()
            return True

    async def shutdown(self) -> None:
        async with self._slot_lock:
            self._cancel_idle_unload_locked()
            await self._unload_locked()

    async def _unload_locked(self) -> None:
        self._cancel_idle_unload_locked()
        self._loaded_config = None
        self._model = None
        self._tokenizer = None
        self._loaded_at = None
        self._idle_expires_at = None
        gc.collect()
        self._clear_cache()
        delete_loaded_model_state()

    def _current_memory_bytes(self) -> int:
        try:
            return self._process_factory().memory_info().rss
        except psutil.Error:
            return 0

    def _refresh_idle_deadline_locked(self) -> None:
        if self._keep_alive_seconds == 0:
            self._idle_expires_at = None
        else:
            self._idle_expires_at = time.time() + self._keep_alive_seconds

    def _write_runtime_state_locked(self) -> None:
        if self._loaded_config is None:
            delete_loaded_model_state()
            return
        write_loaded_model_state(
            LoadedModelState(
                name=self._loaded_config.name,
                model_id=self._loaded_config.model_id,
                memory_bytes=self._current_memory_bytes(),
                idle_expires_at=self._idle_expires_at,
            )
        )

    def _schedule_idle_unload_locked(self) -> None:
        self._cancel_idle_unload_locked()
        if self._loaded_config is None or self._keep_alive_seconds == 0:
            return
        self._idle_task = asyncio.create_task(self._idle_unload_after(self._keep_alive_seconds))

    def _cancel_idle_unload_locked(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None

    async def _idle_unload_after(self, delay_seconds: int) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            async with self._slot_lock:
                if self._inference_lock.locked():
                    self._schedule_idle_unload_locked()
                    return
                await self._unload_locked()
        except asyncio.CancelledError:
            return
