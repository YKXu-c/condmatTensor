"""Band structure analysis and plotting."""

from typing import Optional, List, Tuple
import torch
import math

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

    def plot_with_dos(
        self,
        eigenvalues_mesh: torch.Tensor,
        omega: torch.Tensor,
        eta: float = 0.02,
        energy_range: Optional[Tuple[float, float]] = None,
        ylabel: str = r"Energy ($|t|$)",
        dos_xlabel: str = r"DOS (states/$|t|$)",
        title: str = "Band Structure + DOS",
        fontsize: int = 12,
        figsize: Tuple[float, float] = (10, 5),
        dos_color: str = "skyblue",
        **kwargs,
    ) -> Tuple["maxes.Axes", "maxes.Axes"]:
        """
        Plot band structure with DOS (shared y-axis).

        Creates a two-panel plot:
        - Left: Band structure along k-path
        - Right: DOS calculated from full k-mesh eigenvalues

        The y-axis (energy) is shared between both panels.

        Args:
            eigenvalues_mesh: Eigenvalues on full k-mesh, shape (N_k_mesh, N_band)
                             Used for DOS calculation
            omega: Energy grid for DOS, shape (n_omega,)
            eta: Lorentzian broadening width for DOS
            energy_range: (ymin, ymax) for energy axis
            ylabel: Label for y-axis (shared)
            dos_xlabel: Label for DOS x-axis
            title: Plot title
            fontsize: Font size for labels
            figsize: Figure size (width, height)
            dos_color: Color for DOS fill
            **kwargs: Additional arguments for band plot()

        Returns:
            (ax_bands, ax_dos) tuple of matplotlib axes

        Example:
            >>> # Band structure along k-path
            >>> k_path = generate_k_path(lattice, ['G', 'K', 'M', 'G'], 100)
            >>> Hk_path = Hr.to_k_space(k_path)
            >>> E_path = diagonalize(Hk_path)[0]
            >>>
            >>> # DOS from full k-mesh
            >>> k_mesh = generate_kmesh(lattice, 50)
            >>> Hk_mesh = Hr.to_k_space(k_mesh)
            >>> E_mesh = diagonalize(Hk_mesh)[0]
            >>> omega = torch.linspace(-4, 4, 500)
            >>>
            >>> # Combined plot
            >>> bs = BandStructure()
            >>> bs.compute(E_path, k_path, ticks=[(0, 'G'), (33, 'K'), (66, 'M'), (99, 'G')])
            >>> bs.plot_with_dos(E_mesh, omega, eta=0.05)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        if self.eigenvalues is None:
            raise ValueError("No eigenvalues stored. Call compute() first.")

        # Create figure with two subplots sharing y-axis
        fig, (ax_bands, ax_dos) = plt.subplots(
            1, 2, figsize=figsize, sharey=True, gridspec_kw={'width_ratios': [3, 1]}
        )

        N_k, N_band = self.eigenvalues.shape

        # === Left panel: Band structure ===
        k_dist = torch.arange(N_k).numpy()
        eigenvalues_np = self.eigenvalues.numpy()

        for n in range(N_band):
            ax_bands.plot(k_dist, eigenvalues_np[:, n], **kwargs)

        # Add high-symmetry point markers
        if self.ticks is not None:
            tick_positions = [pos for pos, _ in self.ticks]
            tick_labels = [label for _, label in self.ticks]
            ax_bands.set_xticks(tick_positions)
            ax_bands.set_xticklabels(tick_labels, fontsize=fontsize)
            for pos in tick_positions:
                ax_bands.axvline(x=pos, color="gray", linestyle="--", alpha=0.3)

        ax_bands.set_xlabel("k-path", fontsize=fontsize)
        ax_bands.set_ylabel(ylabel, fontsize=fontsize)
        ax_bands.grid(True, alpha=0.3)
        ax_bands.tick_params(labelleft=True)  # Show y-tick labels on left

        # === Right panel: DOS ===
        # Calculate DOS from eigenvalues on full k-mesh
        N_k_mesh, N_band_mesh = eigenvalues_mesh.shape
        eps_flat = eigenvalues_mesh.flatten()  # (N_k_mesh * N_band,)

        # Create broadcasting grids
        omega_grid = omega[:, None]  # (n_omega, 1)
        eps_grid = eps_flat[None, :]  # (1, N_k_mesh * N_band)

        # Lorentzian: (η/π) / [(ω - ε)² + η²]
        lorentzian = (eta / math.pi) / ((omega_grid - eps_grid) ** 2 + eta ** 2)

        # DOS = (1/N_k) Σ over all states
        rho = torch.sum(lorentzian, dim=1) / N_k_mesh

        # Convert to numpy for matplotlib
        omega_np = omega.cpu().numpy()
        rho_np = rho.cpu().numpy()

        # Plot DOS (horizontal: x = DOS, y = energy)
        ax_dos.fill_betweenx(omega_np, 0, rho_np, color=dos_color, alpha=0.7)
        ax_dos.plot(rho_np, omega_np, color='black', linewidth=1)
        ax_dos.set_xlabel(dos_xlabel, fontsize=fontsize)
        ax_dos.grid(True, alpha=0.3, axis='x')

        # Remove y-tick labels on right (shared with left)
        ax_dos.tick_params(labelleft=False)

        # Set energy range
        if energy_range is not None:
            ax_bands.set_ylim(energy_range)
            ax_dos.set_ylim(energy_range)

        # Set title
        fig.suptitle(title, fontsize=fontsize + 2)
        fig.tight_layout()

        return ax_bands, ax_dos
