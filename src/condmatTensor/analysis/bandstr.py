"""Band structure analysis and plotting."""

from typing import Optional, List, Tuple
import torch

try:
    import matplotlib.pyplot as plt
    import matplotlib.axes as maxes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class BandStructure:
    """
    Band structure calculator with plotting.

    Stores eigenvalues along high-symmetry paths and
    provides visualization.
    """

    def __init__(self) -> None:
        """Initialize BandStructure calculator."""
        self.k_path: Optional[torch.Tensor] = None
        self.eigenvalues: Optional[torch.Tensor] = None
        self.ticks: Optional[List[Tuple[int, str]]] = None

    def compute(
        self,
        eigenvalues: torch.Tensor,
        k_path: torch.Tensor,
        ticks: Optional[List[Tuple[int, str]]] = None,
    ) -> None:
        """
        Store band structure results.

        Args:
            eigenvalues: Eigenvalues at each k-point, shape (N_k, N_band)
            k_path: K-point path in fractional coordinates, shape (N_k, dim)
            ticks: List of (index, label) for high-symmetry point markers
        """
        self.eigenvalues = eigenvalues
        self.k_path = k_path
        self.ticks = ticks

    def plot(
        self,
        ax: Optional["maxes.Axes"] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        ylabel: str = r"Energy $t$",
        title: str = "Band Structure",
        fontsize: int = 12,
        **kwargs,
    ) -> "maxes.Axes":
        """
        Plot band structure.

        Args:
            ax: Matplotlib axis (if None, creates new figure)
            energy_range: (ymin, ymax) for energy axis
            ylabel: Label for y-axis
            title: Plot title
            fontsize: Font size for labels
            **kwargs: Additional arguments for plot()

        Returns:
            Matplotlib axis with band structure plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        if self.eigenvalues is None:
            raise ValueError("No eigenvalues stored. Call compute() first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        N_k, N_band = self.eigenvalues.shape

        # Convert to numpy for matplotlib
        k_dist = torch.arange(N_k).numpy()
        eigenvalues_np = self.eigenvalues.numpy()

        # Plot each band
        for n in range(N_band):
            ax.plot(k_dist, eigenvalues_np[:, n], **kwargs)

        # Add high-symmetry point markers
        if self.ticks is not None:
            tick_positions = [pos for pos, _ in self.ticks]
            tick_labels = [label for _, label in self.ticks]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=fontsize)
            # Add vertical lines at high-symmetry points
            for pos in tick_positions:
                ax.axvline(x=pos, color="gray", linestyle="--", alpha=0.3)

        ax.set_xlabel("k-path", fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.grid(True, alpha=0.3)

        if energy_range is not None:
            ax.set_ylim(energy_range)

        return ax
