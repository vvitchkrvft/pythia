# Pythia

Pythia is an Ollama-compatible API server and process manager for MLX models on Apple Silicon. It starts and supervises multiple `mlx_lm.server` processes from a single `config.yaml`, then exposes a simple Ollama-style API in front of them. The goal is straightforward: keep the operational model of `ollama serve`, but use native MLX backends instead of `llama.cpp`, which is generally a better fit for M-series machines and makes multi-model local serving practical.

## Requirements

- Apple Silicon Mac
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)

## Installation

Install Pythia as a CLI tool with `uv`:

```bash
uv tool install .
```

If you are working from a checkout and want to run it without installing globally:

```bash
uv run pythia --help
```

## Quick Start

Create a `config.yaml`:

```yaml
models:
  - name: qwen3-coder
    model_id: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
    port: 8080
  - name: glm-flash
    model_id: mlx-community/GLM-4.7-Flash-4bit
    port: 8081
```

Start everything:

```bash
pythia serve
```

This does two things:

- Starts one `mlx_lm.server` process per model in `config.yaml`
- Starts the Ollama-compatible API server on `127.0.0.1:11434`

List models:

```bash
curl http://127.0.0.1:11434/api/tags
```

Chat with a model:

```bash
curl http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder",
    "messages": [
      {"role": "user", "content": "Write a Python function that reverses a string."}
    ]
  }'
```

## Configuration

Pythia reads `config.yaml` from the current working directory by default.

Format:

```yaml
models:
  - name: qwen3-coder
    model_id: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
    port: 8080
  - name: glm-flash
    model_id: mlx-community/GLM-4.7-Flash-4bit
    port: 8081
```

Fields:

- `name`: local model alias used by the CLI and API
- `model_id`: Hugging Face repo ID for the MLX model
- `port`: local port for that model's `mlx_lm.server` process

## CLI

### `pythia serve`

Starts all configured model servers and the integrated API server. The command blocks until interrupted.

Options:

- `--config`, `-c`: path to `config.yaml`
- `--api-port`: API port to bind, default `11434`

On shutdown, Pythia stops the model processes it started.

### `pythia ps`

Reads `~/.pythia/pids/`, verifies tracked processes, and prints a Rich table with:

- model name
- port
- PID
- status
- RAM usage

### `pythia stop`

Stops tracked model processes and removes their PID files.

Examples:

```bash
pythia stop qwen3-coder
pythia stop --all
```

## API

Pythia exposes an Ollama-compatible API on `127.0.0.1:11434` by default.

Implemented endpoints:

- `GET /api/tags`
- `POST /api/pull`
- `POST /api/generate`
- `POST /api/chat`
- `DELETE /api/delete`
- `POST /api/show`

Notes:

- The API uses the configured model alias from `config.yaml` as the model identifier.
- `generate` and `chat` proxy requests to the corresponding `mlx_lm.server` instance for that alias.
- `delete` stops the model and removes its Hugging Face cache entry.
- `show` follows the Ollama HTTP contract and expects a JSON body such as `{"name": "qwen3-coder"}`.

## State and Logs

Pythia stores runtime state under `~/.pythia/`:

- `~/.pythia/pids/`: tracked process metadata
- `~/.pythia/logs/`: stderr logs for model server startup failures

## Roadmap

- `pythia pull` CLI command
- OpenAI-compatible `/v1` API layer
- GGUF to MLX conversion workflow
