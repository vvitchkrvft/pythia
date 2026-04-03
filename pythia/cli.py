from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from pythia.registry import ModelRegistry
from pythia.server import create_app
from pythia.state import (
    delete_api_pid,
    delete_loaded_model_state,
    read_api_server_status,
    read_loaded_model_state,
    remaining_idle_seconds,
    stop_api_server,
    write_api_pid,
)

app = typer.Typer(help="Pythia API server and in-process model manager.")
console = Console()


def _warn(message: str) -> None:
    console.print(f"[yellow]Warning:[/yellow] {message}")


def _format_ram(ram_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(ram_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{ram_bytes} B"


def _format_idle_time(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    return f"{int(max(0, seconds))}s"


def _print_pull_status(payload: dict[str, object]) -> None:
    status = str(payload.get("status", ""))
    model_name = payload.get("model")
    error = payload.get("error")

    if status == "error":
        if isinstance(error, str) and error:
            console.print(f"[red]error[/red]: {error}")
        else:
            console.print("[red]error[/red]")
        return

    if model_name:
        console.print(f"{status}: {model_name}")
        return

    console.print(status)


@app.command()
def serve(
    config: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        "-c",
        exists=False,
        dir_okay=False,
        help="Path to config.yaml",
    ),
    api_port: int = typer.Option(
        11434,
        "--api-port",
        help="Port for the integrated Pythia API server",
    ),
) -> None:
    """Start the integrated Pythia API server."""
    try:
        ModelRegistry(config)
        write_api_pid(os.getpid(), host="127.0.0.1", port=api_port)
        console.print(f"Starting API server on 127.0.0.1:{api_port}")
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(config),
                host="127.0.0.1",
                port=api_port,
                log_level="info",
            )
        )
        server.run()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(code=1) from error
    except KeyboardInterrupt:
        console.print("\nShutting down Pythia...")
    finally:
        delete_api_pid()
        delete_loaded_model_state()


@app.command("ps")
def ps_command() -> None:
    """Show API server status and the currently loaded model."""
    api_status = read_api_server_status(warn=_warn)
    api_table = Table(title="API Server")
    api_table.add_column("Status")
    api_table.add_column("PID", justify="right")
    api_table.add_column("Host")
    api_table.add_column("Port", justify="right")
    api_style = "green" if api_status.status == "running" else "red"
    api_table.add_row(
        f"[{api_style}]{api_status.status}[/{api_style}]",
        str(api_status.pid) if api_status.pid is not None else "-",
        api_status.host,
        str(api_status.port),
    )
    console.print(api_table)

    if api_status.status != "running":
        delete_loaded_model_state()

    loaded_model = read_loaded_model_state(warn=_warn) if api_status.status == "running" else None
    if loaded_model is None:
        console.print("No model currently loaded.")
        return

    table = Table(title="Loaded Model")
    table.add_column("Name")
    table.add_column("Model ID")
    table.add_column("Memory", justify="right")
    table.add_column("Idle Unload", justify="right")
    table.add_row(
        loaded_model.name,
        loaded_model.model_id,
        _format_ram(loaded_model.memory_bytes),
        _format_idle_time(remaining_idle_seconds(loaded_model)),
    )
    console.print(table)


@app.command()
def stop() -> None:
    """Stop the Pythia API server."""
    if stop_api_server(warn=_warn):
        console.print("Stopped Pythia API server.")
    else:
        console.print("No running Pythia API server found.")


@app.command()
def pull(
    model_name: str = typer.Argument(..., metavar="MODEL_NAME"),
    config: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        "-c",
        exists=False,
        dir_okay=False,
        help="Path to config.yaml",
    ),
    api_port: int = typer.Option(
        11434,
        "--api-port",
        help="Port for the integrated Pythia API server",
    ),
) -> None:
    """Pull a model through the running Pythia API server."""
    _ = config
    url = f"http://127.0.0.1:{api_port}/api/pull"

    try:
        with httpx.stream("POST", url, json={"model": model_name}, timeout=None) as response:
            if response.status_code == 404:
                try:
                    error_detail = response.json().get("detail", "")
                except (json.JSONDecodeError, ValueError, AttributeError):
                    error_detail = response.text
                fallback_error = f"Unknown model '{model_name}'"
                console.print(f"[red]Error:[/red] {error_detail or fallback_error}")
                raise typer.Exit(code=1)

            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                _print_pull_status(payload)
                if payload.get("status") == "success":
                    console.print(f"Pulled {model_name} successfully.")
                    return
                if payload.get("status") == "error":
                    raise typer.Exit(code=1)

            console.print("[red]Error:[/red] Pull did not complete successfully.")
            raise typer.Exit(code=1)
    except httpx.RequestError as error:
        console.print("Pythia API server is not running. Start it with 'pythia serve'.")
        raise typer.Exit(code=1) from error
    except httpx.HTTPStatusError as error:
        console.print(f"[red]Error:[/red] {error.response.text or 'Request failed.'}")
        raise typer.Exit(code=1) from error
