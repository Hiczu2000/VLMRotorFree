from rich.console import Console
console = Console()

def log(msg:str, 
        level:str="INFO", 
        shift_number: int = 0, 
        color:bool = False):
    """
    Print the log communicate in console

    Parameters
    ----------
    msg : str
        Message to print
    level : str, optional
        Level defines the icon printed next to 
        the message and the color of it,
        the avaible levels are:
        "INFO"  ·,
        "OK"    ✓,
        "WARN"  ⚠,
        "ERR"   ✗,
        "STEP"  ▶, 
        by default "INFO"
    shift_number : int, optional
        Number of the tabs before the 
        console communicate, 
        by default 0
    color : bool, optional
        If True, plot the communicate 
        with related colors, 
        by default False
    """
    icons = {
        "INFO": "·",
        "OK": "✓",
        "WARN": "⚠",
        "ERR": "✗",
        "STEP": "▶"
    }

    text = "  " * shift_number + f"  {icons.get(level, '·')} {msg}"

    if color:
        if level == "INFO":
            clr = "yellow3"
        elif level =="OK":
            clr = "bright_green"
        elif level =="WARN":
            clr = "orange3"
        console.print(text, style=clr)
    else:
        print(text)


def section(title:str):
    """
    Print section division

    Parameters
    ----------
    title : str
        Section title
    """
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def summary(name:str, 
            params: dict):
    """
    Print the table with the names and 
    values stored in the input dictionary

    Parameters
    ----------
    name : str
        Name of summary
    params : dict
        Names and values of the considered properties
    """
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    for k, v in params.items():
        if isinstance(v, int):
            console.print(f"  {k:<20} {v:.0f}", style = "yellow3")
        elif isinstance(v, float):
            console.print(f"  {k:<20} {v:.4f}", style = "yellow3")
        else:
            console.print(f"  {k:<20} {v}", style = "yellow3")
    print(f"{'─'*60}\n")