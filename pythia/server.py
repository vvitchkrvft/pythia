from __future__ import annotations

import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from pythia.config import ModelConfig, load_config
from pythia.process import list_process_statuses, stop_model


class PullRequest(BaseModel):
    model: str


class DeleteRequest(BaseModel):
    model: str


def create_app(config_path: Path = Path("config.yaml")) -> FastAPI:
    app = FastAPI(title="Pythia API")

    def load_models() -> list[ModelConfig]:
        return load_config(config_path)

    def get_model(name: str) -> ModelConfig:
        for model in load_models():
            if model.name == name:
                return model
        raise HTTPException(status_code=404, detail=f"Unknown model '{name}'")

    def get_status_map() -> dict[str, str]:
        return {status.name: status.status for status in list_process_statuses()}

    def get_cached_model_path(model_id: str) -> Path | None:
        try:
            return Path(snapshot_download(repo_id=model_id, local_files_only=True))
        except Exception:
            return None

    def get_model_stats(model_id: str) -> tuple[int, str | None]:
        model_path = get_cached_model_path(model_id)
        if model_path is None or not model_path.exists():
            return 0, None

        total_size = 0
        latest_mtime = 0.0
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                total_size += stat.st_size
                latest_mtime = max(latest_mtime, stat.st_mtime)

        modified_at = None
        if latest_mtime:
            modified_at = datetime.fromtimestamp(latest_mtime, tz=UTC).isoformat()
        return total_size, modified_at

    def now_iso() -> str:
        return datetime.now(tz=UTC).isoformat()

    def ollama_line(payload: dict[str, Any]) -> bytes:
        return (json.dumps(payload) + "\n").encode("utf-8")

    async def proxy_json(
        model: ModelConfig,
        path: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"http://127.0.0.1:{model.port}{path}",
                json=payload,
            )
            response.raise_for_status()
            return response

    async def iter_openai_stream(response: httpx.Response) -> Any:
        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            yield json.loads(data)

    def ensure_running(model: ModelConfig) -> None:
        if get_status_map().get(model.name) != "running":
            raise HTTPException(
                status_code=409,
                detail=f"Model '{model.name}' is not running. Start it with `pythia serve`.",
            )

    @app.get("/api/tags")
    async def api_tags() -> dict[str, list[dict[str, Any]]]:
        status_map = get_status_map()
        models: list[dict[str, Any]] = []
        for model in load_models():
            size, modified_at = get_model_stats(model.model_id)
            models.append(
                {
                    "name": model.name,
                    "model": model.model_id,
                    "status": status_map.get(model.name, "stopped"),
                    "size": size,
                    "modified_at": modified_at,
                }
            )
        return {"models": models}

    @app.post("/api/pull")
    async def api_pull(request: PullRequest) -> StreamingResponse:
        model = get_model(request.model)

        async def event_stream() -> Any:
            yield ollama_line({"status": "pulling manifest", "model": model.name})
            yield ollama_line({"status": "downloading", "model": model.name})
            try:
                await asyncio.to_thread(snapshot_download, repo_id=model.model_id)
                size, _modified_at = get_model_stats(model.model_id)
                yield ollama_line(
                    {
                        "status": "success",
                        "model": model.name,
                        "completed": size,
                        "total": size,
                    }
                )
            except Exception as error:
                yield ollama_line(
                    {
                        "status": "error",
                        "model": model.name,
                        "error": str(error),
                    }
                )

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @app.post("/api/generate")
    async def api_generate(payload: dict[str, Any]) -> Any:
        model_name = payload.get("model")
        prompt = payload.get("prompt")
        if not isinstance(model_name, str) or not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="Expected string 'model' and 'prompt'")

        model = get_model(model_name)
        ensure_running(model)
        stream = bool(payload.get("stream", False))
        backend_payload = {
            "prompt": prompt,
            "stream": stream,
        }

        if stream:
            async def event_stream() -> Any:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        f"http://127.0.0.1:{model.port}/v1/completions",
                        json=backend_payload,
                    ) as response:
                        response.raise_for_status()
                        async for chunk in iter_openai_stream(response):
                            choice = chunk["choices"][0]
                            text = choice.get("text", "")
                            done = choice.get("finish_reason") is not None
                            yield ollama_line(
                                {
                                    "model": model.name,
                                    "created_at": now_iso(),
                                    "response": text,
                                    "done": done,
                                }
                            )

            return StreamingResponse(event_stream(), media_type="application/x-ndjson")

        try:
            response = await proxy_json(model, "/v1/completions", backend_payload)
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

        data = response.json()
        choice = data["choices"][0]
        return JSONResponse(
            {
                "model": model.name,
                "created_at": now_iso(),
                "response": choice.get("text", ""),
                "done": choice.get("finish_reason") is not None,
            }
        )

    @app.post("/api/chat")
    async def api_chat(payload: dict[str, Any]) -> Any:
        model_name = payload.get("model")
        messages = payload.get("messages")
        if not isinstance(model_name, str) or not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="Expected 'model' and 'messages'")

        model = get_model(model_name)
        ensure_running(model)
        stream = bool(payload.get("stream", False))
        backend_payload = {
            "messages": messages,
            "stream": stream,
        }

        if stream:
            async def event_stream() -> Any:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        f"http://127.0.0.1:{model.port}/v1/chat/completions",
                        json=backend_payload,
                    ) as response:
                        response.raise_for_status()
                        async for chunk in iter_openai_stream(response):
                            choice = chunk["choices"][0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            done = choice.get("finish_reason") is not None
                            yield ollama_line(
                                {
                                    "model": model.name,
                                    "created_at": now_iso(),
                                    "message": {
                                        "role": delta.get("role", "assistant"),
                                        "content": content,
                                    },
                                    "done": done,
                                }
                            )

            return StreamingResponse(event_stream(), media_type="application/x-ndjson")

        try:
            response = await proxy_json(model, "/v1/chat/completions", backend_payload)
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message", {})
        return JSONResponse(
            {
                "model": model.name,
                "created_at": now_iso(),
                "message": {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content", ""),
                },
                "done": choice.get("finish_reason") is not None,
            }
        )

    @app.delete("/api/delete")
    async def api_delete(request: DeleteRequest) -> dict[str, Any]:
        model = get_model(request.model)
        stop_result = stop_model(model.name)

        cached_path = get_cached_model_path(model.model_id)
        removed_path = False
        if cached_path is not None and cached_path.exists():
            shutil.rmtree(cached_path, ignore_errors=True)
            removed_path = True

        return {
            "status": "success",
            "model": model.name,
            "stopped": bool(stop_result and stop_result.was_running),
            "removed": removed_path,
        }

    @app.get("/api/show")
    async def api_show(model: str = Query(...)) -> dict[str, Any]:
        model_config = get_model(model)
        size, modified_at = get_model_stats(model_config.model_id)
        status = get_status_map().get(model_config.name, "stopped")
        return {
            "name": model_config.name,
            "model": model_config.model_id,
            "port": model_config.port,
            "status": status,
            "size": size,
            "modified_at": modified_at,
        }

    return app
