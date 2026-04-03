from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from pythia.config import load_config
from pythia.process import (
    list_process_statuses,
    load_record,
    start_model_server,
    stop_all_models,
    stop_model,
)

app = typer.Typer(help="Pythia model process manager.")
console = Console()


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
    )
) -> None:
    """Start one mlx_lm.server process per configured model."""
    try:
        models = load_config(config)
        for model in models:
            record = start_model_server(model)
            console.print(
                f"Started {record.name} on port {record.port} with PID {record.pid}"
            )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(code=1) from error


@app.command("ps")
def ps_command() -> None:
    """Show model processes tracked by Pythia."""
    statuses = list_process_statuses()
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
    model_name: Annotated[str | None, typer.Argument(help="Model name to stop")] = None,
    all: bool = typer.Option(False, "--all", help="Stop all tracked models"),
) -> None:
    """Stop one tracked model or all tracked models."""
    if all:
        stop_results = stop_all_models()
        if not stop_results:
            console.print("No tracked models found in ~/.pythia/pids/")
            return

        for result in stop_results:
            if result.was_running:
                console.print(f"Stopped {result.record.name} (PID {result.record.pid})")
            else:
                console.print(f"{result.record.name} was already stopped.")
        return

    if model_name is None:
        console.print("Provide a model name or use --all.")
        return

    record = load_record(model_name)
    if record is None:
        console.print(f"No tracked model named '{model_name}' found.")
        return

    stop_result = stop_model(model_name)
    if stop_result is None:
        console.print(f"No tracked model named '{model_name}' found.")
        return

    if stop_result.was_running:
        console.print(f"Stopped {stop_result.record.name} (PID {stop_result.record.pid})")
    else:
        console.print(f"{stop_result.record.name} is already stopped.")
