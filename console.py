from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import sys
from multiprocessing import active_children

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down...")
    # Terminate all child processes
    for process in active_children():
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
    sys.exit(0)

def console_process(log_queue):
    """Dedicated process for all console output"""
    try:
        console = Console()

        while True:
            msg = log_queue.get()

            if msg['type'] == 'meter':
                # Use console's native method to update same line
                console.print(msg['text'], end="\r", markup=True)

            elif msg['type'] == 'user':
                # Print newline to clear meter, then show panel
                console.print()  # Clear meter line
                console.print(Panel(f"[bold green]{msg['text']}[/bold green]",
                                    title="🧑 You",
                                    border_style="green"))

            elif msg['type'] == 'ai':
                console.print()
                console.print(Panel(Markdown(msg['text']),
                                    title="🤖 AI",
                                    border_style="cyan",
                                    padding=(1, 2)))

            elif msg['type'] == 'info':
                console.print()
                console.print(msg['text'])

            elif msg['type'] == 'status':
                console.print()
                console.print(f"[dim]{msg['text']}[/dim]")

            elif msg['type'] == 'error':
                console.print()
                console.print(f"[red]❌ {msg['text']}[/red]")
    except KeyboardInterrupt:
        print("LOG shutting down...")
    finally:
        pass