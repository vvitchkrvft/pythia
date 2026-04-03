from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from pythia.config import load_config
from pythia.process import (
    delete_api_pid,
    list_process_statuses,
    load_record,
    stop_api_server,
    stop_all_models,
    stop_model,
    write_api_pid,
)
from pythia.server import create_app

app = typer.Typer(help="Pythia API server and model process manager.")
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
        load_config(config)
        write_api_pid(os.getpid())
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
        stop_results = stop_all_models(warn=_warn)
        for result in stop_results:
            if result.was_running:
                console.print(f"Stopped {result.record.name} (PID {result.record.pid})")


@app.command("ps")
def ps_command() -> None:
    """Show model processes tracked by Pythia."""
    statuses = list_process_statuses(warn=_warn)
    if not statuses:
        console.print("No tracked processes found in ~/.pythia/pids/")
        return

    table = Table(title="Pythia Processes")
    table.add_column("Model Name")
    table.add_column("Port", justify="right")
    table.add_column("PID", justify="right")
    table.add_column("Status")
    table.add_column("RAM", justify="right")

    for status in statuses:
        status_style = "green" if status.status == "running" else "red"
        table.add_row(
            status.name,
            str(status.port),
            str(status.pid),
            f"[{status_style}]{status.status}[/{status_style}]",
            _format_ram(status.ram_bytes),
        )

    console.print(table)


@app.command()
def stop(
    model_name: Annotated[
        str | None, typer.Argument(help="Model name to stop. Omit to stop the API server.")
    ] = None,
    all: bool = typer.Option(False, "--all", help="Stop the API server and all tracked models"),
) -> None:
    """Stop the API server, or stop one model by name."""
    if all:
        api_stopped = stop_api_server(warn=_warn)
        stop_results = stop_all_models(warn=_warn)
        if api_stopped:
            console.print("Stopped Pythia API server.")

        for result in stop_results:
            if result.was_running:
                console.print(f"Stopped {result.record.name} (PID {result.record.pid})")
            else:
                console.print(f"{result.record.name} was already stopped.")
        if not api_stopped and not stop_results:
            console.print("No running Pythia API server or tracked models found.")
        return

    if model_name is None:
        if stop_api_server(warn=_warn):
            console.print("Stopped Pythia API server.")
        else:
            console.print("No running Pythia API server found.")
        return

    record = load_record(model_name, warn=_warn)
    if record is None:
        console.print(f"No tracked model named '{model_name}' found.")
        return

    stop_result = stop_model(model_name, warn=_warn)
    if stop_result is None:
        console.print(f"No tracked model named '{model_name}' found.")
        return

    if stop_result.was_running:
        console.print(f"Stopped {stop_result.record.name} (PID {stop_result.record.pid})")
    else:
        console.print(f"{stop_result.record.name} is already stopped.")
