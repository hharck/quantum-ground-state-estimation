from matplotlib.axes import Axes

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def style(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the house style to an axis: left-aligned title, faint y grid, no top/right spines.

    Args:
        ax: the axis to restyle, modified in place.
        title: chart title, placed at the left edge.
        xlabel: label under the x axis.
        ylabel: label beside the y axis.

    Returns:
        Nothing; `ax` is changed in place.
    """
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def phase_axes(ax: Axes, n: int) -> None:
    """Label an axis with every n-bit readout outcome, and the phase estimate it stands for.

    The measured bits j run along the bottom as binary strings; a second axis on top shows
    the same positions as the estimate j / 2^n.

    Args:
        ax: the axis to label, modified in place.
        n: number of measured bits, so the axis gets 2**n ticks.

    Returns:
        Nothing; `ax` is changed in place.
    """
    N = 2**n
    ax.set_xticks(range(N), [f"{j:0{n}b}" for j in range(N)], rotation=90)
    top = ax.secondary_xaxis("top", functions=(lambda x: x / N, lambda y: y * N))
    top.set_xticks(
        [j / N for j in range(N)], [f"{j / N:.4f}" for j in range(N)], rotation=90
    )
    top.set_xlabel(r"estimated $\varphi = j / 2^n$")
