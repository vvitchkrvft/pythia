from __future__ import annotations

import asyncio
import threading
import time
import unittest

import httpx

from pythia.config import ModelConfig
from pythia.process import ProcessRecord
from pythia.runtime import CHAT_PROBE_PAYLOAD, RuntimeManager


def make_record(model: ModelConfig, pid: int = 1234) -> ProcessRecord:
    return ProcessRecord(
        name=model.name,
        model_id=model.model_id,
        port=model.port,
        pid=pid,
        create_time=1.0,
        cmdline=["python", "-m", "mlx_lm.server"],
    )


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://127.0.0.1")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)

    def json(self) -> object:
        return self._payload


class FakeAsyncClient:
    def __init__(self, script: dict[str, object], order: list[tuple[str, str, object | None]]) -> None:
        self._script = script
        self._order = order

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str) -> FakeResponse:
        self._order.append(("GET", url, None))
        next_health = self._script["health"]
        if callable(next_health):
            status_code = next_health()
        else:
            status_code = next_health
        return FakeResponse(status_code, {})

    async def post(self, url: str, json: object) -> FakeResponse:
        self._order.append(("POST", url, json))
        next_probe = self._script["probe"]
        if callable(next_probe):
            status_code, payload = next_probe()
        else:
            status_code, payload = next_probe
        return FakeResponse(status_code, payload)


class RuntimeManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.model = ModelConfig(
            name="alpha",
            model_id="mlx-community/test-model",
            port=8080,
        )

    async def test_single_flight_startup_for_concurrent_requests(self) -> None:
        records: dict[str, ProcessRecord] = {}
        start_calls = 0
        order: list[tuple[str, str, object | None]] = []
        counter_lock = threading.Lock()

        def start_server(model: ModelConfig) -> ProcessRecord:
            nonlocal start_calls
            with counter_lock:
                start_calls += 1
            time.sleep(0.05)
            record = make_record(model)
            records[model.name] = record
            return record

        def live_record_loader(alias: str) -> ProcessRecord | None:
            return records.get(alias)

        manager = RuntimeManager(
            [self.model],
            start_server=start_server,
            live_record_loader=live_record_loader,
            async_client_factory=lambda **_: FakeAsyncClient(
                {
                    "health": 200,
                    "probe": (
                        200,
                        {"choices": [{"message": {"role": "assistant", "content": "pong"}}]},
                    ),
                },
                order,
            ),
        )

        await asyncio.gather(
            manager.ensure_ready(self.model.name),
            manager.ensure_ready(self.model.name),
        )

        self.assertEqual(start_calls, 1)
        self.assertEqual(manager.get_status(self.model.name).state, "ready")
        self.assertEqual(order.count(("GET", f"http://127.0.0.1:{self.model.port}/health", None)), 1)

    async def test_failed_startup_clears_in_flight_state(self) -> None:
        start_calls = 0
        records: dict[str, ProcessRecord] = {}

        def failing_start_server(model: ModelConfig) -> ProcessRecord:
            nonlocal start_calls
            start_calls += 1
            raise RuntimeError("boom")

        def successful_start_server(model: ModelConfig) -> ProcessRecord:
            nonlocal start_calls
            start_calls += 1
            record = make_record(model)
            records[model.name] = record
            return record

        def live_record_loader(alias: str) -> ProcessRecord | None:
            return records.get(alias)

        manager = RuntimeManager(
            [self.model],
            start_server=failing_start_server,
            live_record_loader=live_record_loader,
            async_client_factory=lambda **_: FakeAsyncClient(
                {
                    "health": 200,
                    "probe": (
                        200,
                        {"choices": [{"message": {"role": "assistant", "content": "pong"}}]},
                    ),
                },
                [],
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await manager.ensure_ready(self.model.name)

        self.assertIsNone(manager._entries[self.model.name].startup_task)
        self.assertEqual(manager.get_status(self.model.name).state, "failed")

        manager._start_server = successful_start_server
        await manager.ensure_ready(self.model.name)

        self.assertEqual(start_calls, 2)
        self.assertEqual(manager.get_status(self.model.name).state, "ready")

    async def test_readiness_probe_success_path(self) -> None:
        records: dict[str, ProcessRecord] = {}
        order: list[tuple[str, str, object | None]] = []
        health_statuses = [503, 200]

        def start_server(model: ModelConfig) -> ProcessRecord:
            record = make_record(model)
            records[model.name] = record
            return record

        def live_record_loader(alias: str) -> ProcessRecord | None:
            return records.get(alias)

        def next_health() -> int:
            return health_statuses.pop(0)

        manager = RuntimeManager(
            [self.model],
            start_server=start_server,
            live_record_loader=live_record_loader,
            poll_interval_seconds=0.0,
            async_client_factory=lambda **_: FakeAsyncClient(
                {
                    "health": next_health,
                    "probe": (
                        200,
                        {"choices": [{"message": {"role": "assistant", "content": "pong"}}]},
                    ),
                },
                order,
            ),
        )

        await manager.ensure_ready(self.model.name)

        self.assertEqual(manager.get_status(self.model.name).state, "ready")
        self.assertEqual(
            order,
            [
                ("GET", f"http://127.0.0.1:{self.model.port}/health", None),
                ("GET", f"http://127.0.0.1:{self.model.port}/health", None),
                (
                    "POST",
                    f"http://127.0.0.1:{self.model.port}/v1/chat/completions",
                    CHAT_PROBE_PAYLOAD,
                ),
            ],
        )

    async def test_readiness_probe_failure_path(self) -> None:
        records: dict[str, ProcessRecord] = {}
        stop_calls: list[str] = []

        def start_server(model: ModelConfig) -> ProcessRecord:
            record = make_record(model)
            records[model.name] = record
            return record

        def stop_server(alias: str) -> None:
            stop_calls.append(alias)
            records.pop(alias, None)

        def live_record_loader(alias: str) -> ProcessRecord | None:
            return records.get(alias)

        manager = RuntimeManager(
            [self.model],
            start_server=start_server,
            stop_server=stop_server,
            live_record_loader=live_record_loader,
            startup_timeout_seconds=0.02,
            poll_interval_seconds=0.0,
            async_client_factory=lambda **_: FakeAsyncClient(
                {"health": 200, "probe": (200, {"choices": [{}]})},
                [],
            ),
        )

        with self.assertRaisesRegex(TimeoutError, "Timed out waiting for model"):
            await manager.ensure_ready(self.model.name)

        self.assertIsNone(manager._entries[self.model.name].startup_task)
        status = manager.get_status(self.model.name)
        self.assertEqual(status.state, "failed")
        self.assertIn("Timed out", status.last_error or "")
        self.assertEqual(stop_calls, [self.model.name])


if __name__ == "__main__":
    unittest.main()
