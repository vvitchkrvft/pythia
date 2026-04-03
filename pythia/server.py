from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import scan_cache_dir, snapshot_download
from mlx_lm import generate, stream_generate
from pydantic import BaseModel

from pythia.registry import ModelRegistry
from pythia.runtime import RuntimeManager
from pythia.state import read_loaded_model_state


class PullRequest(BaseModel):
    model: str


class DeleteRequest(BaseModel):
    model: str


class ShowRequest(BaseModel):
    name: str


def create_app(config_path: Path = Path("config.yaml")) -> FastAPI:
    registry = ModelRegistry(config_path)
    runtime = RuntimeManager(registry)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await runtime.shutdown()

    app = FastAPI(title="Pythia API", lifespan=lifespan)

    def get_model(name: str):
        model = registry.get(name)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Unknown model '{name}'")
        return model

    def get_cached_model_path(model_id: str) -> Path | None:
        try:
            return Path(snapshot_download(repo_id=model_id, local_files_only=True))
        except Exception:
            return None

    def delete_cached_model(model_id: str) -> bool:
        try:
            cache_info = scan_cache_dir()
        except Exception:
            return False

        repo = next((repo for repo in cache_info.repos if repo.repo_id == model_id), None)
        if repo is None:
            return False

        revision_hashes = [revision.commit_hash for revision in repo.revisions]
        if not revision_hashes:
            return False

        delete_strategy = cache_info.delete_revisions(*revision_hashes)
        delete_strategy.execute()
        return True

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

    def sse_line(payload: dict[str, Any] | str) -> bytes:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        return f"data: {data}\n\n".encode("utf-8")

    def chat_prompt(tokenizer: object, messages: list[dict[str, Any]]) -> str:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def openai_chat_id() -> str:
        return f"chatcmpl-{uuid4().hex}"

    def openai_created() -> int:
        return int(datetime.now(tz=UTC).timestamp())

    def openai_model_row(model_name: str) -> dict[str, str]:
        return {"id": model_name, "object": "model", "owned_by": "pythia"}

    def generation_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
        options = payload.get("options", {})
        max_tokens = payload.get("max_tokens")
        temperature = payload.get("temperature")
        if isinstance(options, dict):
            if max_tokens is None:
                max_tokens = options.get("num_predict")
            if temperature is None:
                temperature = options.get("temperature")

        kwargs: dict[str, Any] = {}
        if isinstance(max_tokens, int):
            kwargs["max_tokens"] = max_tokens
        if isinstance(temperature, int | float):
            kwargs["temp"] = float(temperature)
        return kwargs

    async def run_generate(model: object, tokenizer: object, prompt: str, kwargs: dict[str, Any]) -> str:
        return await asyncio.to_thread(generate, model, tokenizer, prompt, **kwargs)

    async def iter_generate(model: object, tokenizer: object, prompt: str, kwargs: dict[str, Any]):
        iterator = stream_generate(model, tokenizer, prompt, **kwargs)

        def next_chunk():
            try:
                return next(iterator)
            except StopIteration:
                return None

        while True:
            chunk = await asyncio.to_thread(next_chunk)
            if chunk is None:
                break
            yield chunk

    async def chat_response_text(model_name: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
        async with runtime.session(model_name) as session:
            prompt = chat_prompt(session.tokenizer, messages)
            return await run_generate(session.model, session.tokenizer, prompt, kwargs)

    async def chat_chunks(model_name: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]):
        async with runtime.session(model_name) as session:
            prompt = chat_prompt(session.tokenizer, messages)
            async for chunk in iter_generate(session.model, session.tokenizer, prompt, kwargs):
                yield chunk

    @app.get("/api/tags")
    async def api_tags() -> dict[str, list[dict[str, Any]]]:
        current_model = runtime.current_model() or read_loaded_model_state()
        model_rows: list[dict[str, Any]] = []
        for model in registry.all():
            size, modified_at = get_model_stats(model.model_id)
            status = "loaded" if current_model and current_model.name == model.name else "available"
            model_rows.append(
                {
                    "name": model.name,
                    "model": model.name,
                    "status": status,
                    "size": size,
                    "modified_at": modified_at,
                }
            )
        return {"models": model_rows}

    @app.post("/api/pull")
    async def api_pull(request: PullRequest) -> StreamingResponse:
        model = get_model(request.model)

        async def event_stream():
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

        get_model(model_name)
        stream = bool(payload.get("stream", False))
        kwargs = generation_kwargs(payload)

        if stream:
            needs_load = runtime.needs_load(model_name)

            async def event_stream():
                if needs_load:
                    yield ollama_line(
                        {
                            "model": model_name,
                            "created_at": now_iso(),
                            "response": "",
                            "done": False,
                            "status": "loading model",
                        }
                    )
                async with runtime.session(model_name) as session:
                    async for chunk in iter_generate(session.model, session.tokenizer, prompt, kwargs):
                        yield ollama_line(
                            {
                                "model": model_name,
                                "created_at": now_iso(),
                                "response": chunk.text,
                                "done": chunk.finish_reason is not None,
                            }
                        )

            return StreamingResponse(event_stream(), media_type="application/x-ndjson")

        async with runtime.session(model_name) as session:
            response_text = await run_generate(session.model, session.tokenizer, prompt, kwargs)
        return JSONResponse(
            {
                "model": model_name,
                "created_at": now_iso(),
                "response": response_text,
                "done": True,
            }
        )

    @app.post("/api/chat")
    async def api_chat(payload: dict[str, Any]) -> Any:
        model_name = payload.get("model")
        messages = payload.get("messages")
        if not isinstance(model_name, str) or not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="Expected 'model' and 'messages'")

        get_model(model_name)
        stream = bool(payload.get("stream", False))
        kwargs = generation_kwargs(payload)

        if stream:
            needs_load = runtime.needs_load(model_name)

            async def event_stream():
                if needs_load:
                    yield ollama_line(
                        {
                            "model": model_name,
                            "created_at": now_iso(),
                            "message": {"role": "assistant", "content": ""},
                            "done": False,
                            "status": "loading model",
                        }
                    )
                async for chunk in chat_chunks(model_name, messages, kwargs):
                    yield ollama_line(
                        {
                            "model": model_name,
                            "created_at": now_iso(),
                            "message": {"role": "assistant", "content": chunk.text},
                            "done": chunk.finish_reason is not None,
                        }
                    )

            return StreamingResponse(event_stream(), media_type="application/x-ndjson")

        response_text = await chat_response_text(model_name, messages, kwargs)
        return JSONResponse(
            {
                "model": model_name,
                "created_at": now_iso(),
                "message": {"role": "assistant", "content": response_text},
                "done": True,
            }
        )

    @app.delete("/api/delete")
    async def api_delete(request: DeleteRequest) -> dict[str, Any]:
        model = get_model(request.model)
        unloaded = await runtime.unload(model.name)
        removed_path = delete_cached_model(model.model_id)

        return {
            "status": "success",
            "model": model.name,
            "stopped": unloaded,
            "removed": removed_path,
        }

    @app.post("/api/show")
    async def api_show(request: ShowRequest) -> dict[str, Any]:
        model_config = get_model(request.name)
        size, modified_at = get_model_stats(model_config.model_id)
        current_model = runtime.current_model()
        status = "loaded" if current_model and current_model.name == model_config.name else "available"
        return {
            "name": model_config.name,
            "model": model_config.name,
            "status": status,
            "size": size,
            "modified_at": modified_at,
            "details": {"model_id": model_config.model_id},
        }

    @app.get("/v1/models")
    async def openai_models() -> dict[str, Any]:
        return {"object": "list", "data": [openai_model_row(model.name) for model in registry.all()]}

    @app.get("/v1/models/{model_ref:path}")
    async def openai_model(model_ref: str) -> dict[str, str]:
        model = get_model(model_ref)
        return openai_model_row(model.name)

    async def openai_chat(payload: dict[str, Any]) -> Any:
        model_name = payload.get("model")
        messages = payload.get("messages")
        if not isinstance(model_name, str) or not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="Expected 'model' and 'messages'")

        model = get_model(model_name)
        stream = bool(payload.get("stream", False))
        kwargs = generation_kwargs(payload)
        completion_id = openai_chat_id()
        created = openai_created()

        if stream:
            async def event_stream():
                async for chunk in chat_chunks(model.name, messages, kwargs):
                    finish_reason = chunk.finish_reason
                    delta: dict[str, Any] = {}
                    if chunk.text:
                        delta["content"] = chunk.text
                    yield sse_line(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model.name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                    )
                yield sse_line("[DONE]")

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        response_text = await chat_response_text(model.name, messages, kwargs)
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model.name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def openai_chat_v1(payload: dict[str, Any]) -> Any:
        return await openai_chat(payload)

    @app.post("/chat/completions")
    async def openai_chat_unversioned(payload: dict[str, Any]) -> Any:
        return await openai_chat(payload)

    return app
