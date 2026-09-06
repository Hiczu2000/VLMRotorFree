import matplotlib.pyplot as plt


def apply_thesis_style():
    """Apply consistent matplotlib styling (fonts, sizes, grid) for thesis figures."""

    plt.rcParams.update({

        # figure
        "figure.figsize": (6, 4),
        "figure.dpi": 120,
        "savefig.dpi": 300,

        # fonts
        "font.family": "serif",
        "font.size": 16,

        # axes
        "axes.labelsize": 16,
        "axes.titlesize": 18,

        # ticks
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,

        # legend
        "legend.fontsize": 14,

        # lines
        "lines.linewidth": 2.2,
        "lines.markersize": 6,

        # grid
        "axes.grid": True,
        "grid.alpha": 0.3,

        # cleaner axes
        "axes.spines.top": False,
        "axes.spines.right": False,

        # export
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


LINE_COLORS = [
    "black",     # primary
    "#1811f0e2",   
    "#118f26",   
    "#942de9",   
    "#da0606ef",   # dark red
    "#1adaebfb",   # teal
    "#f08b06",   # dark orange
    "#ff0066f3" ,
    "#5eef0f",   # bright blue  
    "#ffcc00" ,
    "#95999b",   # gray
    "#78107af3" ,
    "#ff4400f3" ,
    "#763406f3" ,
]

POINT_COLORS = [
    "#d62828",   # vivid red
    "#f77f00",   # vivid orange
    "#ffcc00",   # vivid yellow
]


def thesis_plot(ax,
                x,
                y,
                label=None,
                xlabel=None,
                ylabel=None,
                title=None,
                style="line",
                color=None):
    """Plot a line or point series on ax with consistent thesis styling and auto-cycled colors."""

    plotted_lines = [
        line for line in ax.lines
        if line.get_linestyle() != "None"
    ]

    plotted_points = [
        line for line in ax.lines
        if line.get_linestyle() == "None"
    ]

    if style == "line":
        idx = len(plotted_lines)

        if color is None:
            color = LINE_COLORS[idx % len(LINE_COLORS)]

        ax.plot(
            x,
            y,
            linestyle="--",
            color=color,
            linewidth=2.2,
            label=label
        )

    elif style == "point":
        idx = len(plotted_points)

        if color is None:
            color = POINT_COLORS[idx % len(POINT_COLORS)]

        ax.plot(
            x,
            y,
            marker="o",
            linestyle="None",
            color=color,
            markersize=8,
            label=label
        )

    else:
        raise ValueError(f"Unknown plot style: {style}")

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)


def save_figure(fig,
                name,
                folder="figures"):
    """Save figure as both PDF and high-resolution PNG in the given folder."""

    fig.savefig(f"{folder}/{name}.pdf")
    fig.savefig(f"{folder}/{name}.png", dpi=300)