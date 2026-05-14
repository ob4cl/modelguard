"""ModelGuard CLI — command-line interface for scanning models.

Usage:
    modelguard scan path/to/model.safetensors
    modelguard scan --hub meta-llama/Llama-2-7b-hf
    modelguard scan --activations --behavioral path/to/model/
    modelguard scan --format sarif path/to/model/
    modelguard hub audit --top 10
    modelguard sbom path/to/model/
    modelguard signatures list
    modelguard signatures search "badnets"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .sbom.generator import SBOMGenerator
from .scanner import scan_hub, scan_model
from .types import Severity

console = Console()


def _severity_style(severity: Severity) -> str:
    """Map severity to Rich style."""
    return {
        Severity.CRITICAL: "bold red",
        Severity.HIGH: "red",
        Severity.MEDIUM: "yellow",
        Severity.LOW: "dim",
        Severity.INFO: "blue",
    }[severity]


def _severity_emoji(severity: Severity) -> str:
    return {
        Severity.CRITICAL: "🔴",
        Severity.HIGH: "🟠",
        Severity.MEDIUM: "🟡",
        Severity.LOW: "🔵",
        Severity.INFO: "⚪",
    }[severity]


@click.group()
@click.version_option(__version__, prog_name="modelguard")
def main() -> None:
    """ModelGuard — ML supply chain security scanner.

    Detect backdoors, poisoned layers, and adversarial triggers
    in model weights. Like trivy for neural networks.
    """


@main.command()
@click.argument("model_path")
@click.option(
    "--hub", is_flag=True, help="MODEL_PATH is a HuggingFace Hub repo ID"
)
@click.option(
    "--token", default=None, help="HuggingFace API token for gated models"
)
@click.option(
    "--format", "output_format", default="table",
    type=click.Choice(["table", "json", "sarif"]),
    help="Output format",
)
@click.option(
    "--weights/--no-weights", default=True,
    help="Enable/disable weight anomaly scanning",
)
@click.option(
    "--signatures/--no-signatures", default=True,
    help="Enable/disable signature matching",
)
@click.option(
    "--activations", is_flag=True,
    help="Enable activation pattern scanning (requires torch+transformers)",
)
@click.option(
    "--behavioral", is_flag=True,
    help="Enable behavioral trigger testing (requires torch+transformers)",
)
@click.option(
    "--fail-on", default="high",
    type=click.Choice(["critical", "high", "medium", "low", "never"]),
    help="Exit with non-zero code if findings at or above this severity",
)
def scan(
    model_path: str,
    hub: bool,
    token: str | None,
    output_format: str,
    weights: bool,
    signatures: bool,
    activations: bool,
    behavioral: bool,
    fail_on: str,
) -> None:
    """Scan a model for backdoors and supply chain risks.

    MODEL_PATH can be a local file, directory, or HuggingFace repo ID (with --hub).
    """
    with console.status(f"[bold]Scanning {model_path}...[/bold]"):
        if hub:
            result = scan_hub(
                model_path, token=token, weights=weights, signatures=signatures,
                activations=activations, behavioral=behavioral,
            )
        else:
            result = scan_model(
                model_path, weights=weights, signatures=signatures,
                activations=activations, behavioral=behavioral,
            )

    if output_format == "json":
        console.print(result.to_json())
    elif output_format == "sarif":
        console.print(_to_sarif(result))
    else:
        _print_table(result)

    # Exit code based on severity threshold
    severity_levels = {
        "critical": 0, "high": 1, "medium": 2, "low": 3, "never": 99,
    }
    threshold = severity_levels[fail_on]
    max_severity = 0
    for f in result.findings:
        sv = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
        max_severity = max(max_severity, sv.get(f.severity.value, 4))

    if max_severity <= threshold and threshold < 99:
        sys.exit(1)


@main.command()
@click.argument("model_path")
def sbom(model_path: str) -> None:
    """Generate a Software Bill of Materials for a model."""
    with console.status(f"[bold]Generating SBOM for {model_path}...[/bold]"):
        generator = SBOMGenerator()
        result = generator.generate(Path(model_path))

    console.print_json(result)


@main.group()
def hub() -> None:
    """HuggingFace Hub operations."""


@hub.command("audit")
@click.option(
    "--top", default=5, type=int,
    help="Number of trending models to audit",
)
@click.option(
    "--token", default=None, help="HuggingFace API token for gated models",
)
@click.option(
    "--format", "output_format", default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
def hub_audit(top: int, token: str | None, output_format: str) -> None:
    """Audit trending HuggingFace models for backdoors."""
    from huggingface_hub import list_models

    console.print(f"[bold]Auditing top {top} trending models on HuggingFace...[/bold]\n")

    models = list(list_models(sort="downloads", direction=-1, limit=top))

    results = []
    for model in models:
        console.print(f"  Scanning [cyan]{model.id}[/cyan]...")
        try:
            result = scan_hub(model.id, token=token)
            results.append((model.id, result))
        except Exception as e:
            console.print(f"    [red]Failed: {e}[/red]")

    if output_format == "json":
        output = {
            "audited": [
                {"model_id": mid, "result": r.to_dict()}
                for mid, r in results
            ]
        }
        console.print_json(json.dumps(output, indent=2))
    else:
        # Summary table
        table = Table(title="Hub Audit Results")
        table.add_column("Model")
        table.add_column("Status")
        table.add_column("Critical")
        table.add_column("High")
        table.add_column("Medium")
        table.add_column("Total")

        for model_id, result in results:
            d = result.to_dict()["summary"]
            status = "✅" if result.passed else "❌"
            table.add_row(
                model_id,
                status,
                str(d["critical"]),
                str(d["high"]),
                str(d["medium"]),
                str(d["total"]),
            )

        console.print(table)


@main.group()
def signatures() -> None:
    """Manage backdoor signature database."""


@signatures.command("list")
def signatures_list() -> None:
    """List all known backdoor signatures."""
    from .signatures.known_backdoors import KNOWN_BACKDOORS

    table = Table(title="Known Backdoor Signatures")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Severity")
    table.add_column("Reference")

    for entry in KNOWN_BACKDOORS:
        severity_style = {
            "critical": "bold red", "high": "red",
            "medium": "yellow", "low": "dim",
        }.get(entry.severity, "")
        table.add_row(
            entry.id,
            entry.name,
            f"[{severity_style}]{entry.severity}[/{severity_style}]",
            entry.reference[:60] + "..." if len(entry.reference) > 60 else entry.reference,
        )

    console.print(table)


@signatures.command("search")
@click.argument("query")
def signatures_search(query: str) -> None:
    """Search known backdoor signatures."""
    from .signatures.known_backdoors import KNOWN_BACKDOORS

    query_lower = query.lower()
    results = [
        e for e in KNOWN_BACKDOORS
        if query_lower in e.name.lower()
        or query_lower in e.description.lower()
        or query_lower in e.id.lower()
    ]

    if not results:
        console.print(f"[yellow]No signatures matching '{query}'[/yellow]")
        return

    for entry in results:
        console.print(
            Panel(
                f"[bold]{entry.name}[/bold]\n"
                f"ID: {entry.id}\n"
                f"Severity: {entry.severity}\n\n"
                f"{entry.description}\n\n"
                f"Reference: {entry.reference}\n"
                f"Mitigation: {entry.mitigation or 'None specified'}",
                title=entry.id,
                border_style=_severity_style(
                    Severity(entry.severity.upper())
                ),
            )
        )


def _print_table(result) -> None:  # type: ignore
    """Print scan results as a Rich table."""

    # Summary panel
    if result.passed:
        summary = Text("✅ PASSED — No critical or high severity findings", style="green")
    else:
        summary = Text(
            f"❌ FAILED — {result.critical_count} critical, "
            f"{result.high_count} high severity findings",
            style="bold red",
        )

    console.print(
        Panel(
            f"[bold]Model:[/bold] {result.model_path}\n"
            f"[bold]Hash:[/bold]  {result.model_hash[:32]}...\n"
            f"[bold]Time:[/bold]  {result.duration_ms:.0f}ms\n\n"
            f"{summary}\n\n"
            f"Total findings: {len(result.findings)}",
            title="ModelGuard Scan Results",
            border_style="green" if result.passed else "red",
        )
    )

    # Findings table
    if result.findings:
        table = Table(title="Findings")
        table.add_column("Severity", style="bold")
        table.add_column("Rule ID")
        table.add_column("Message")
        table.add_column("Layer")

        for finding in result.findings:
            table.add_row(
                f"{_severity_emoji(finding.severity)} {finding.severity.value}",
                finding.rule_id,
                finding.message[:80] + "..." if len(finding.message) > 80 else finding.message,
                finding.layer_name or "-",
            )

        console.print(table)
        console.print(
            "\n[dim]Run with --format json for machine-readable output[/dim]"
        )


def _to_sarif(result) -> str:  # type: ignore
    """Convert scan result to SARIF format."""
    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "ModelGuard",
                        "version": __version__,
                        "informationUri": "https://github.com/ob4cl/modelguard",
                    }
                },
                "results": [
                    {
                        "ruleId": f.rule_id,
                        "level": f.severity.value.lower(),
                        "message": {"text": f.message},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": result.model_path,
                                    }
                                }
                            }
                        ],
                    }
                    for f in result.findings
                ],
            }
        ],
    }
    return json.dumps(sarif, indent=2)
