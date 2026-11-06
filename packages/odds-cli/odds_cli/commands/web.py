"""CLI commands for the read-only web UI."""

import typer
from rich.console import Console

app = typer.Typer()
console = Console()

# The tailnet reaches the app through `tailscale serve`, so the app itself stays on loopback.
LOOPBACK = "127.0.0.1"


@app.command("serve")
def serve(port: int = typer.Option(8765, "--port", help="Port to listen on")) -> None:
    """Serve the web UI on 127.0.0.1 as the read-only WEB_DATABASE_URL role."""
    # Imported here so the scheduler's `odds` invocations do not load the web stack.
    import uvicorn
    from odds_web.app import create_app_from_settings
    from odds_web.db import WebDatabaseNotConfigured

    try:
        web_app = create_app_from_settings()
    except WebDatabaseNotConfigured as exc:
        console.print(f"[red]{exc}[/red]", soft_wrap=True)
        raise typer.Exit(1) from exc
    uvicorn.run(web_app, host=LOOPBACK, port=port)
