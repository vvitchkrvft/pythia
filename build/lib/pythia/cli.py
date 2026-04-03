from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pythia.config import load_config
from pythia.process import list_process_statuses, start_model_server

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
