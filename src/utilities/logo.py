from pyfiglet import Figlet
from rich.console import Console

def print_logo(text:str):
    """
    Print the logo with the solver name

    Parameters
    ----------
    text : str
        Name of the solver
    """
    console = Console()
    f = Figlet(font="standard")
    logo = f.renderText(text)
    console.print(logo, style="bold blue")