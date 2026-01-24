"""Band structure analysis and plotting."""

from typing import Optional, List, Tuple, Union
import torch
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.axes as maxes
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from condmatTensor.analysis.plotting_style import (
    DEFAULT_FIGURE_SIZES,
    DEFAULT_COLORS,
    DEFAULT_FONTSIZES,
    DEFAULT_STYLING,
    DEFAULT_COLORMAPS,
    LINE_STYLES,
)


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

    def plot_colored_by_weight(
        self,
        eigenvectors: torch.Tensor,
        orbital_indices: Union[List[int], Tuple[int, ...]],
        ax: Optional["maxes.Axes"] = None,
        cmap: str = "viridis",
        s: float = 10.0,
        vmin: float = 0.0,
        vmax: float = 1.0,
        ylabel: str = r"Energy ($|t|$)",
        xlabel: str = "k-path",
        title: str = "Band Structure (colored by orbital weight)",
        fontsize: int = 12,
        colorbar: bool = True,
        colorbar_label: str = "Orbital weight",
        **kwargs,
    ) -> "maxes.Axes":
        """
        Plot band structure colored by orbital weight using scatter plot.

        Each point in the band structure is colored by the weight of the
        specified orbitals in that eigenstate. Useful for visualizing
        orbital character of bands.

        Args:
            eigenvectors: Eigenvectors at each k-point, shape (N_k, N_orb, N_band)
            orbital_indices: Indices of orbitals to compute weight for
            ax: Matplotlib axis (if None, creates new figure)
            cmap: Colormap name (default: 'viridis')
            s: Marker size for scatter plot
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            ylabel: Label for y-axis
            xlabel: Label for x-axis
            title: Plot title
            fontsize: Font size for labels
            colorbar: Whether to show colorbar
            colorbar_label: Label for colorbar
            **kwargs: Additional arguments for scatter()

        Returns:
            Matplotlib axis with colored band structure plot

        Example:
            >>> # Color by f-orbital character (orbitals 6,7)
            >>> bs = BandStructure()
            >>> bs.compute(eigenvalues, k_path, ticks)
            >>> bs.plot_colored_by_weight(eigenvectors, orbital_indices=[6, 7])
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        if self.eigenvalues is None:
            raise ValueError("No eigenvalues stored. Call compute() first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZES['single'])

        N_k, N_band = self.eigenvalues.shape
        k_dist = torch.arange(N_k)

        # Compute orbital weight for each (k, band) pair
        weight = torch.zeros((N_k, N_band))
        for k in range(N_k):
            for band in range(N_band):
                # Sum of squared amplitudes on specified orbitals
                weight[k, band] = torch.sum(
                    torch.abs(eigenvectors[k, orbital_indices, band]) ** 2
                )

        # Convert to numpy for matplotlib
        k_np = k_dist.numpy()
        eig_np = self.eigenvalues.numpy()
        weight_np = weight.numpy()

        # Plot each band with scatter, colored by weight
        for band in range(N_band):
            ax.scatter(
                k_np,
                eig_np[:, band],
                c=weight_np[:, band],
                cmap=cmap,
                s=s,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )

        # Add high-symmetry point markers
        if self.ticks is not None:
            tick_positions = [pos for pos, _ in self.ticks]
            tick_labels = [label for _, label in self.ticks]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=fontsize)
            for pos in tick_positions:
                ax.axvline(x=pos, color="gray", linestyle="--", alpha=DEFAULT_STYLING['grid_alpha'])

        ax.set_xlabel(xlabel, fontsize=DEFAULT_FONTSIZES['labels'])
        ax.set_ylabel(ylabel, fontsize=DEFAULT_FONTSIZES['labels'])
        ax.set_title(title, fontsize=DEFAULT_FONTSIZES['titles'])
        ax.grid(True, alpha=DEFAULT_STYLING['grid_alpha'])

        # Add colorbar
        if colorbar:
            scatter = ax.scatter(
                k_np[[0]], eig_np[[0, 0], 0], c=weight_np[[0, 0], 0],
                cmap=cmap, vmin=vmin, vmax=vmax, s=0
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(colorbar_label, fontsize=DEFAULT_FONTSIZES['labels'])

        return ax

    def add_reference_line(
        self,
        energy: float,
        label: Optional[str] = None,
        color: str = "red",
        linestyle: str = "--",
        alpha: float = 0.7,
        ax: Optional["maxes.Axes"] = None,
        **kwargs,
    ) -> None:
        """
        Add horizontal reference line to existing plot.

        Useful for marking flat bands, Fermi level, or other energy references.

        Args:
            energy: Energy value for horizontal line
            label: Label for the line (shown in legend)
            color: Line color (default: 'red')
            linestyle: Line style ('-', '--', ':', '-.')
            alpha: Line transparency
            ax: Matplotlib axis (if None, uses current axis)
            **kwargs: Additional arguments for axhline()

        Example:
            >>> bs = BandStructure()
            >>> bs.compute(eigenvalues, k_path, ticks)
            >>> ax = bs.plot()
            >>> bs.add_reference_line(-2.0, label='Flat band')
            >>> plt.legend()
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        if ax is None:
            ax = plt.gca()

        ax.axhline(
            y=energy,
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            label=label,
            **kwargs,
        )

    def plot_comparison(
        self,
        other_eigenvalues: Union[torch.Tensor, List[torch.Tensor]],
        labels: Union[str, List[str]],
        colors: Optional[Union[str, List[str]]] = None,
        alpha: float = 0.6,
        ax: Optional["maxes.Axes"] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        ylabel: str = r"Energy ($|t|$)",
        title: str = "Band Structure Comparison",
        fontsize: int = 12,
        linewidth: float = 1.0,
        legend: bool = True,
        **kwargs,
    ) -> "maxes.Axes":
        """
        Overlay multiple band structures for comparison.

        Useful for comparing full vs effective models, or parameter sweeps.

        Args:
            other_eigenvalues: Eigenvalues to compare.
                Single tensor: plots this vs stored eigenvalues.
                List of tensors: plots all of them.
            labels: Label(s) for the legend
            colors: Color(s) for the plots (default: auto-generated)
            alpha: Transparency for overlaid lines
            ax: Matplotlib axis (if None, creates new figure)
            energy_range: (ymin, ymax) for energy axis
            ylabel: Label for y-axis
            title: Plot title
            fontsize: Font size for labels
            linewidth: Line width
            legend: Whether to show legend
            **kwargs: Additional arguments for plot()

        Returns:
            Matplotlib axis with comparison plot

        Example:
            >>> # Compare full vs effective model
            >>> bs = BandStructure()
            >>> bs.compute(eigenvalues_full, k_path, ticks)
            >>> bs.plot_comparison(eigenvalues_eff, labels=['Full', 'Effective'])
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        if self.eigenvalues is None:
            raise ValueError("No eigenvalues stored. Call compute() first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZES['comparison_2'])

        # Normalize input to lists
        if isinstance(other_eigenvalues, torch.Tensor):
            other_eigenvalues = [other_eigenvalues]
        if isinstance(labels, str):
            labels = [labels]

        # Default colors
        if colors is None:
            default_colors = [DEFAULT_COLORS['primary'], DEFAULT_COLORS['secondary'],
                            DEFAULT_COLORS['tertiary'], DEFAULT_COLORS['quaternary']]
            colors = default_colors[:len(other_eigenvalues) + 1]
        if isinstance(colors, str):
            colors = [colors]

        # Plot all band structures
        all_eigenvalues = [self.eigenvalues] + other_eigenvalues
        all_labels = ['Original'] + list(labels)
        all_colors = list(colors) if len(colors) > 1 else [colors[0]] * len(all_eigenvalues)

        N_k = self.eigenvalues.shape[0]

        for eig, label, color in zip(all_eigenvalues, all_labels, all_colors):
            k_dist = torch.arange(N_k).numpy()
            eig_np = eig.cpu().numpy() if hasattr(eig, 'cpu') else eig.numpy()
            N_band = eig.shape[1]

            for n in range(N_band):
                ax.plot(k_dist, eig_np[:, n], color=color, alpha=alpha,
                       linewidth=linewidth, label=label if n == 0 else "", **kwargs)

        # Add high-symmetry point markers
        if self.ticks is not None:
            tick_positions = [pos for pos, _ in self.ticks]
            tick_labels = [label for _, label in self.ticks]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=fontsize)
            for pos in tick_positions:
                ax.axvline(x=pos, color="gray", linestyle="--", alpha=DEFAULT_STYLING['grid_alpha'])

        ax.set_xlabel("k-path", fontsize=DEFAULT_FONTSIZES['labels'])
        ax.set_ylabel(ylabel, fontsize=DEFAULT_FONTSIZES['labels'])
        ax.set_title(title, fontsize=DEFAULT_FONTSIZES['titles'])
        ax.grid(True, alpha=DEFAULT_STYLING['grid_alpha'])

        if energy_range is not None:
            ax.set_ylim(energy_range)

        if legend:
            ax.legend(fontsize=DEFAULT_FONTSIZES['legend'])

        return ax

    def plot_multi_panel(
        self,
        eigenvalues_list: List[torch.Tensor],
        titles: List[str],
        k_paths: Optional[List[torch.Tensor]] = None,
        ticks_list: Optional[List[List[Tuple[int, str]]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        ylabel: str = r"Energy ($|t|$)",
        fontsize: int = 12,
        **kwargs,
    ) -> Tuple["maxes.Axes", ...]:
        """
        Create multi-panel band structure comparison.

        Each panel shows a separate band structure. Useful for comparing
        different parameters, models, or systems.

        Args:
            eigenvalues_list: List of eigenvalues, each shape (N_k, N_band)
            titles: Title for each panel
            k_paths: Optional list of k-paths (if None, uses stored k_path)
            ticks_list: Optional list of tick specifications for each panel
            figsize: Figure size (auto-calculated if None)
            energy_range: (ymin, ymax) for energy axis (shared)
            ylabel: Label for y-axis
            fontsize: Font size for labels
            **kwargs: Additional arguments for plot()

        Returns:
            Tuple of matplotlib axes, one for each panel

        Example:
            >>> # Compare 3 different t_f values
            >>> bs = BandStructure()
            >>> eigenvalues_list = [eig1, eig2, eig3]
            >>> titles = ['t_f = -1.0', 't_f = -0.5', 't_f = 0.0']
            >>> axes = bs.plot_multi_panel(eigenvalues_list, titles)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        n_panels = len(eigenvalues_list)

        # Auto-layout
        if n_panels == 1:
            ncols, nrows = 1, 1
        elif n_panels == 2:
            ncols, nrows = 2, 1
        elif n_panels == 3:
            ncols, nrows = 3, 1
        elif n_panels == 4:
            ncols, nrows = 2, 2
        elif n_panels <= 6:
            ncols, nrows = 3, 2
        else:
            ncols, nrows = 4, 3

        # Auto figsize
        if figsize is None:
            figsize = (ncols * 5, nrows * 4)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_panels == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows * ncols > 1 else [axes]

        # Use stored k_path and ticks if not provided
        if k_paths is None:
            k_paths = [self.k_path] * n_panels
        if ticks_list is None:
            ticks_list = [self.ticks] * n_panels

        for idx, (eig, title, k_path, ticks) in enumerate(
            zip(eigenvalues_list, titles, k_paths, ticks_list)
        ):
            ax = axes[idx]
            N_k, N_band = eig.shape

            # Convert to numpy for matplotlib
            k_dist = torch.arange(N_k).numpy()
            eig_np = eig.numpy() if isinstance(eig, torch.Tensor) else eig

            # Plot each band
            for n in range(N_band):
                ax.plot(k_dist, eig_np[:, n], **kwargs)

            # Add high-symmetry point markers
            if ticks is not None:
                tick_positions = [pos for pos, _ in ticks]
                tick_labels = [label for _, label in ticks]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, fontsize=fontsize)
                for pos in tick_positions:
                    ax.axvline(x=pos, color="gray", linestyle="--",
                              alpha=DEFAULT_STYLING['grid_alpha'])

            ax.set_xlabel("k-path", fontsize=DEFAULT_FONTSIZES['labels'])
            ax.set_ylabel(ylabel, fontsize=DEFAULT_FONTSIZES['labels'])
            ax.set_title(title, fontsize=DEFAULT_FONTSIZES['titles'])
            ax.grid(True, alpha=DEFAULT_STYLING['grid_alpha'])

            if energy_range is not None:
                ax.set_ylim(energy_range)

        # Hide unused subplots
        for idx in range(n_panels, len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()

        return tuple(axes[:n_panels])
