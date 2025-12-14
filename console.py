from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

def console_process(log_queue):
    """Dedicated process for all console output"""
    console = Console()

    while True:
        msg = log_queue.get()

        if msg['type'] == 'user':
            console.print(Panel(f"[bold green]{msg['text']}[/bold green]",
                                title="🧑 You",
                                border_style="green"))

        elif msg['type'] == 'ai':
            console.print(Panel(Markdown(msg['text']),
                                title="🤖 AI",
                                border_style="cyan",
                                padding=(1, 2)))

        elif msg['type'] == 'info':
            console.print(msg['text'])

        elif msg['type'] == 'status':
            console.print(f"[dim]{msg['text']}[/dim]")
