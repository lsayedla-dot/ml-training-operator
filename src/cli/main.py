"""CLI tool for the ML Training Operator."""

from __future__ import annotations

import click
import httpx
from rich.console import Console
from rich.table import Table

console = Console()
DEFAULT_API_URL = "http://localhost:8000"


def _get_client(api_url: str) -> httpx.Client:
    """Create an HTTP client."""
    return httpx.Client(base_url=api_url, timeout=30)


@click.group()
@click.option("--api-url", default=DEFAULT_API_URL, envvar="MLTRAIN_API_URL", help="API server URL")
@click.pass_context
def cli(ctx: click.Context, api_url: str) -> None:
    """ML Training Operator CLI — manage distributed training jobs on Kubernetes."""
    ctx.ensure_object(dict)
    ctx.obj["api_url"] = api_url


@cli.command()
@click.option("--name", required=True, help="Job name")
@click.option(
    "--model",
    "model_type",
    default="resnet18",
    type=click.Choice(["resnet18", "resnet50"]),
)
@click.option("--epochs", default=10, type=int, help="Number of training epochs")
@click.option("--batch-size", default=32, type=int, help="Batch size")
@click.option("--lr", default=0.001, type=float, help="Learning rate")
@click.option("--workers", "num_workers", default=1, type=int, help="Number of DDP workers")
@click.option(
    "--optimize",
    "enable_optimization",
    is_flag=True,
    help="Enable ONNX export + INT8 quantization",
)
@click.option("--cpu", default="2", help="CPU request/limit")
@click.option("--memory", default="4Gi", help="Memory request/limit")
@click.option("--max-retries", default=3, type=int, help="Max retry attempts")
@click.pass_context
def submit(
    ctx: click.Context,
    name: str,
    model_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    enable_optimization: bool,
    cpu: str,
    memory: str,
    max_retries: int,
) -> None:
    """Submit a new training job."""
    payload = {
        "name": name,
        "model_type": model_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_workers": num_workers,
        "enable_optimization": enable_optimization,
        "resources": {"cpu": cpu, "memory": memory},
        "max_retries": max_retries,
    }

    with _get_client(ctx.obj["api_url"]) as client:
        resp = client.post("/jobs", json=payload)
        if resp.status_code == 201:
            job = resp.json()
            console.print("[green]Job submitted successfully![/green]")
            console.print(f"  ID:      {job['id']}")
            console.print(f"  Name:    {job['name']}")
            console.print(f"  Status:  {job['status']}")
            console.print(f"  Workers: {job['num_workers']}")
            if enable_optimization:
                console.print("  Optimization: enabled (ONNX + INT8)")
        else:
            console.print(f"[red]Error: {resp.text}[/red]")


@cli.command("list")
@click.option("--status", default=None, help="Filter by status")
@click.pass_context
def list_jobs(ctx: click.Context, status: str | None) -> None:
    """List all training jobs."""
    params = {}
    if status:
        params["status"] = status

    with _get_client(ctx.obj["api_url"]) as client:
        resp = client.get("/jobs", params=params)
        if resp.status_code != 200:
            console.print(f"[red]Error: {resp.text}[/red]")
            return

        jobs = resp.json()
        if not jobs:
            console.print("[yellow]No jobs found.[/yellow]")
            return

        table = Table(title="Training Jobs")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Status")
        table.add_column("Model")
        table.add_column("Workers")
        table.add_column("Epochs")
        table.add_column("Created")

        for job in jobs:
            status_style = {
                "RUNNING": "green",
                "SUCCEEDED": "bold green",
                "FAILED": "red",
                "DEAD_LETTERED": "bold red",
                "PENDING": "yellow",
                "RETRYING": "yellow",
                "CANCELLED": "dim",
            }.get(job["status"], "")

            table.add_row(
                job["id"],
                job["name"],
                f"[{status_style}]{job['status']}[/{status_style}]",
                job["model_type"],
                str(job["num_workers"]),
                str(job["epochs"]),
                job["created_at"][:19],
            )

        console.print(table)


@cli.command()
@click.argument("job_id")
@click.pass_context
def status(ctx: click.Context, job_id: str) -> None:
    """Get detailed status for a training job."""
    with _get_client(ctx.obj["api_url"]) as client:
        resp = client.get(f"/jobs/{job_id}")
        if resp.status_code == 404:
            console.print(f"[red]Job {job_id} not found[/red]")
            return
        if resp.status_code != 200:
            console.print(f"[red]Error: {resp.text}[/red]")
            return

        job = resp.json()
        console.print(f"[bold]Job: {job['name']}[/bold]")
        console.print(f"  ID:           {job['id']}")
        console.print(f"  Status:       {job['status']}")
        console.print(f"  Model:        {job['model_type']}")
        console.print(f"  Dataset:      {job['dataset']}")
        console.print(f"  Epochs:       {job['epochs']}")
        console.print(f"  Batch Size:   {job['batch_size']}")
        console.print(f"  LR:           {job['learning_rate']}")
        console.print(f"  Workers:      {job['num_workers']}")
        console.print(f"  Optimization: {'enabled' if job['enable_optimization'] else 'disabled'}")
        res = job["resources"]
        console.print(f"  Resources:    {res['cpu']} CPU, {res['memory']} memory")
        console.print(f"  Retries:      {job['retries']}/{job['max_retries']}")
        console.print(f"  Created:      {job['created_at']}")
        if job.get("started_at"):
            console.print(f"  Started:      {job['started_at']}")
        if job.get("completed_at"):
            console.print(f"  Completed:    {job['completed_at']}")
        if job.get("error"):
            console.print(f"  [red]Error:       {job['error']}[/red]")
        if job.get("k8s_job_name"):
            console.print(f"  K8s Job:      {job['k8s_job_name']}")


@cli.command()
@click.argument("job_id")
@click.option("--rank", default=None, type=int, help="Worker rank for distributed jobs")
@click.pass_context
def logs(ctx: click.Context, job_id: str, rank: int | None) -> None:
    """View logs for a training job."""
    # Logs would be fetched from K8s pod logs via the API
    console.print(f"[yellow]Log streaming not yet implemented for job {job_id}[/yellow]")
    if rank is not None:
        console.print(f"  (requested rank {rank})")


@cli.command()
@click.argument("job_id")
@click.pass_context
def cancel(ctx: click.Context, job_id: str) -> None:
    """Cancel a running training job."""
    with _get_client(ctx.obj["api_url"]) as client:
        resp = client.delete(f"/jobs/{job_id}")
        if resp.status_code == 404:
            console.print(f"[red]Job {job_id} not found[/red]")
            return
        if resp.status_code == 200:
            console.print(f"[green]Job {job_id} cancelled successfully[/green]")
        else:
            console.print(f"[red]Error: {resp.text}[/red]")


@cli.command()
@click.option("--max-workers", default=4, type=int, help="Maximum number of workers to benchmark")
@click.pass_context
def benchmark(ctx: click.Context, max_workers: int) -> None:
    """Run a local scaling benchmark with synthetic data."""
    console.print("[bold]Running scaling benchmark...[/bold]")
    console.print(f"  Workers: 1, 2, ... up to {max_workers}")
    console.print("  Using synthetic data (no nuScenes download needed)")

    try:
        from src.worker.benchmark import run_scaling_benchmark

        report = run_scaling_benchmark(max_workers=max_workers)

        table = Table(title="Scaling Benchmark Results")
        table.add_column("Workers")
        table.add_column("Throughput (samples/s)")
        table.add_column("Duration (s)")
        table.add_column("Scaling Efficiency")

        for workers, result in report["worker_results"].items():
            if "error" in result:
                table.add_row(workers, "ERROR", "-", result["error"])
            else:
                efficiency = report["scaling_efficiency"].get(workers, "100.0")
                table.add_row(
                    workers,
                    f"{result['samples_per_second']:.1f}",
                    f"{result['duration_seconds']:.2f}",
                    f"{efficiency}%" if workers != "1" else "baseline",
                )

        console.print(table)
        console.print("[green]Report saved to benchmarks/scaling_report.json[/green]")
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")


if __name__ == "__main__":
    cli()
