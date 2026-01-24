"""Density of States (DOS) calculator with Lorentzian broadening.

Future Work:
    - from_green_function(): Compute DOS from spectral function A(ω) = -(1/π) Im[Gᴿ(ω)]
    - from_spectral_function(): Direct A(ω) input with k-resolved averaging
"""

from typing import Optional, Tuple, List, Union
import torch
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.axes as maxes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from condmatTensor.analysis.plotting_style import (
    DEFAULT_FIGURE_SIZES,
    DEFAULT_COLORS,
    DEFAULT_FONTSIZES,
    DEFAULT_STYLING,
    LINE_STYLES,
)


class DOSCalculator:
    """
    Density of States calculator with Lorentzian broadening.

    Computes DOS from eigenvalues on a k-mesh using:
        ρ(ω) = (1/N_k) Σ_{k,n} (η/π) / [(ω - εₙ(k))² + η²]

    Reference: Standard DOS calculation with Lorentzian broadening
    """

    def __init__(self) -> None:
        """Initialize DOSCalculator."""
        self.omega: Optional[torch.Tensor] = None
        self.rho: Optional[torch.Tensor] = None
        self.eta: Optional[float] = None

    def from_eigenvalues(
        self,
        E_k: torch.Tensor,
        omega: torch.Tensor,
        eta: float = 0.02,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DOS from eigenvalues using Lorentzian broadening.

        Args:
            E_k: Eigenvalues at each k-point, shape (N_k, N_band)
            omega: Energy grid for DOS, shape (n_omega,)
            eta: Lorentzian broadening width (smoothing parameter)

        Returns:
            (omega, rho) tuple - also stored in self.omega, self.rho
        """
        N_k, N_band = E_k.shape

        # Flatten eigenvalues for vectorized computation
        eps_flat = E_k.flatten()  # (N_k * N_band,)

        # Create broadcasting grids
        # omega: (n_omega, 1), eps: (1, N_k * N_band)
        omega_grid = omega[:, None]  # (n_omega, 1)
        eps_grid = eps_flat[None, :]  # (1, N_k * N_band)

        # Lorentzian: (η/π) / [(ω - ε)² + η²]
        lorentzian = (eta / math.pi) / ((omega_grid - eps_grid) ** 2 + eta ** 2)

        # DOS = (1/N_k) Σ over all states
        rho = torch.sum(lorentzian, dim=1) / N_k

        self.omega = omega
        self.rho = rho
        self.eta = eta

        return omega, rho

    def from_spectral_function(
        self,
        A: torch.Tensor,
        omega: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DOS from spectral function A(ω).

        For non-interacting systems, the spectral function equals the DOS.
        For multi-orbital systems, sums over all orbitals:
            DOS(ω) = Σᵢ Aᵢ(ω)

        Args:
            A: Spectral function with shape (n_omega, n_orb)
            omega: Energy grid for DOS (must match A.shape[0])

        Returns:
            (omega, rho) tuple - also stored in self.omega, self.rho
        """
        if A.shape[0] != len(omega):
            raise ValueError(
                f"Spectral function shape {A.shape} doesn't match omega length {len(omega)}"
            )

        # DOS = sum over all orbitals
        rho = torch.sum(A, dim=1)

        self.omega = omega
        self.rho = rho
        self.eta = None  # No broadening parameter when using A(ω)

        return omega, rho

    def plot(
        self,
        ax: Optional["maxes.Axes"] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        ylabel: str = r"DOS (states/$|t|$)",
        xlabel: str = r"Energy $\omega$ ($|t|$)",
        title: str = "Density of States",
        fontsize: int = 12,
        fill: bool = True,
        **kwargs,
    ) -> "maxes.Axes":
        """
        Plot stored DOS results.

        Args:
            ax: Matplotlib axis (if None, creates new figure)
            energy_range: (xmin, xmax) for energy axis
            ylabel: Label for y-axis
            xlabel: Label for x-axis
            title: Plot title
            fontsize: Font size for labels
            fill: Whether to fill under the curve
            **kwargs: Additional arguments for plot()

        Returns:
            Matplotlib axis with DOS plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        if self.omega is None or self.rho is None:
            raise ValueError("Call from_eigenvalues() first to compute DOS")

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        # Convert to numpy for matplotlib
        omega_np = self.omega.cpu().numpy()
        rho_np = self.rho.cpu().numpy()

        # Plot or fill DOS
        if fill:
            ax.fill_between(omega_np, 0, rho_np, **kwargs)
            ax.plot(omega_np, rho_np, color='black', linewidth=1)
        else:
            ax.plot(omega_np, rho_np, **kwargs)

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.grid(True, alpha=0.3)

        if energy_range is not None:
            ax.set_xlim(energy_range)

        return ax

    def plot_with_reference(
        self,
        reference_energies: Union[float, List[float]],
        labels: Optional[Union[str, List[str]]] = None,
        colors: Optional[Union[str, List[str]]] = None,
        linestyles: Optional[Union[str, List[str]]] = None,
        ax: Optional["maxes.Axes"] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        ylabel: str = r"DOS (states/$|t|$)",
        xlabel: str = r"Energy $\omega$ ($|t|$)",
        title: str = "Density of States with Reference Lines",
        fontsize: int = 12,
        legend: bool = True,
        **kwargs,
    ) -> "maxes.Axes":
        """
        Plot DOS with vertical reference lines at specified energies.

        Useful for marking flat bands, Fermi level, Van Hove singularities, etc.

        Args:
            reference_energies: Energy value(s) for vertical reference lines
            labels: Label(s) for the reference lines (shown in legend)
            colors: Color(s) for the reference lines (default: red)
            linestyles: Line style(s) for the reference lines (default: '--')
            ax: Matplotlib axis (if None, creates new figure)
            energy_range: (xmin, xmax) for energy axis
            ylabel: Label for y-axis
            xlabel: Label for x-axis
            title: Plot title
            fontsize: Font size for labels
            legend: Whether to show legend
            **kwargs: Additional arguments for DOS plot (fill, color, etc.)

        Returns:
            Matplotlib axis with DOS plot and reference lines

        Example:
            >>> dos = DOSCalculator()
            >>> dos.from_eigenvalues(eigenvalues, omega, eta=0.05)
            >>> dos.plot_with_reference(-2.0, label='Flat band')
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        # Create base DOS plot
        ax = self.plot(ax=ax, energy_range=energy_range, ylabel=ylabel,
                      xlabel=xlabel, title=title, fontsize=fontsize,
                      color=DEFAULT_COLORS['fill_default'], **kwargs)

        # Normalize inputs to lists
        if isinstance(reference_energies, (int, float)):
            reference_energies = [float(reference_energies)]
        if labels is not None and isinstance(labels, str):
            labels = [labels]
        if colors is not None and isinstance(colors, str):
            colors = [colors]
        if linestyles is not None and isinstance(linestyles, str):
            linestyles = [linestyles]

        # Default styling
        if colors is None:
            colors = [DEFAULT_COLORS['reference']] * len(reference_energies)
        if linestyles is None:
            linestyles = [LINE_STYLES['dashed']] * len(reference_energies)

        # Add reference lines
        for i, energy in enumerate(reference_energies):
            label = labels[i] if labels and i < len(labels) else None
            color = colors[i] if i < len(colors) else colors[-1]
            linestyle = linestyles[i] if i < len(linestyles) else linestyles[-1]

            ax.axvline(x=energy, color=color, linestyle=linestyle,
                      alpha=DEFAULT_STYLING['overlay_alpha'],
                      label=label, linewidth=DEFAULT_STYLING['reference_linewidth'])

        if legend and labels:
            ax.legend(fontsize=DEFAULT_FONTSIZES['legend'])

        return ax

    def plot_comparison(
        self,
        other_dos_data: Union[Tuple[torch.Tensor, torch.Tensor],
                              List[Tuple[torch.Tensor, torch.Tensor]]],
        labels: Union[str, List[str]],
        colors: Optional[Union[str, List[str]]] = None,
        alpha: float = 0.7,
        ax: Optional["maxes.Axes"] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        ylabel: str = r"DOS (states/$|t|$)",
        xlabel: str = r"Energy $\omega$ ($|t|$)",
        title: str = "DOS Comparison",
        fontsize: int = 12,
        linewidth: float = 1.5,
        legend: bool = True,
        fill: bool = False,
        **kwargs,
    ) -> "maxes.Axes":
        """
        Overlay multiple DOS curves for comparison.

        Useful for comparing DOS from different models, parameters, or methods.

        Args:
            other_dos_data: DOS data to compare.
                Tuple (omega, rho): plots this vs stored DOS.
                List of tuples: plots all of them.
            labels: Label(s) for the legend
            colors: Color(s) for the plots (default: auto-generated)
            alpha: Transparency for overlaid curves
            ax: Matplotlib axis (if None, creates new figure)
            energy_range: (xmin, xmax) for energy axis
            ylabel: Label for y-axis
            xlabel: Label for x-axis
            title: Plot title
            fontsize: Font size for labels
            linewidth: Line width
            legend: Whether to show legend
            fill: Whether to fill under curves
            **kwargs: Additional arguments for plot()

        Returns:
            Matplotlib axis with comparison plot

        Example:
            >>> dos1 = DOSCalculator()
            >>> dos1.from_eigenvalues(eig1, omega, eta=0.05)
            >>> dos2 = DOSCalculator()
            >>> dos2.from_eigenvalues(eig2, omega, eta=0.05)
            >>> dos1.plot_comparison((dos2.omega, dos2.rho), labels=['Model 1', 'Model 2'])
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        if self.omega is None or self.rho is None:
            raise ValueError("Call from_eigenvalues() first to compute DOS")

        if ax is None:
            fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZES['single'])

        # Normalize input to lists
        if isinstance(other_dos_data, tuple) and len(other_dos_data) == 2:
            other_dos_data = [other_dos_data]
        if isinstance(labels, str):
            labels = [labels]

        # Default colors
        if colors is None:
            default_colors = [DEFAULT_COLORS['primary'], DEFAULT_COLORS['secondary'],
                            DEFAULT_COLORS['tertiary'], DEFAULT_COLORS['quaternary']]
            colors = default_colors[:len(other_dos_data) + 1]
        if isinstance(colors, str):
            colors = [colors]

        # Collect all DOS data
        all_dos = [(self.omega, self.rho)] + list(other_dos_data)
        all_labels = ['Original'] + list(labels)
        all_colors = list(colors) if len(colors) > 1 else [colors[0]] * len(all_dos)

        for (omega, rho), label, color in zip(all_dos, all_labels, all_colors):
            omega_np = omega.cpu().numpy() if hasattr(omega, 'cpu') else omega.numpy()
            rho_np = rho.cpu().numpy() if hasattr(rho, 'cpu') else rho.numpy()

            if fill:
                ax.fill_between(omega_np, 0, rho_np, color=color, alpha=alpha * 0.5, **kwargs)
                ax.plot(omega_np, rho_np, color=color, linewidth=linewidth, label=label, **kwargs)
            else:
                ax.plot(omega_np, rho_np, color=color, alpha=alpha,
                       linewidth=linewidth, label=label, **kwargs)

        ax.set_xlabel(xlabel, fontsize=DEFAULT_FONTSIZES['labels'])
        ax.set_ylabel(ylabel, fontsize=DEFAULT_FONTSIZES['labels'])
        ax.set_title(title, fontsize=DEFAULT_FONTSIZES['titles'])
        ax.grid(True, alpha=DEFAULT_STYLING['grid_alpha'])

        if energy_range is not None:
            ax.set_xlim(energy_range)

        if legend:
            ax.legend(fontsize=DEFAULT_FONTSIZES['legend'])

        return ax

    def plot_multi_panel(
        self,
        dos_list: List[Tuple[torch.Tensor, torch.Tensor]],
        titles: List[str],
        figsize: Optional[Tuple[float, float]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        ylabel: str = r"DOS (states/$|t|$)",
        xlabel: str = r"Energy $\omega$ ($|t|$)",
        fontsize: int = 12,
        sharex: bool = True,
        sharey: bool = True,
        **kwargs,
    ) -> Tuple["maxes.Axes", ...]:
        """
        Create multi-panel DOS comparison.

        Each panel shows a separate DOS curve. Useful for comparing
        different parameters, models, or systems.

        Args:
            dos_list: List of (omega, rho) tuples for each panel
            titles: Title for each panel
            figsize: Figure size (auto-calculated if None)
            energy_range: (xmin, xmax) for energy axis (shared if sharex=True)
            ylabel: Label for y-axis
            xlabel: Label for x-axis
            fontsize: Font size for labels
            sharex: Whether to share x-axis across panels
            sharey: Whether to share y-axis across panels
            **kwargs: Additional arguments for plot()

        Returns:
            Tuple of matplotlib axes, one for each panel

        Example:
            >>> dos1 = DOSCalculator()
            >>> dos1.from_eigenvalues(eig1, omega, eta=0.05)
            >>> dos2 = DOSCalculator()
            >>> dos2.from_eigenvalues(eig2, omega, eta=0.05)
            >>> axes = dos1.plot_multi_panel(
            ...     [(dos1.omega, dos1.rho), (dos2.omega, dos2.rho)],
            ...     titles=['Model 1', 'Model 2']
            ... )
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        n_panels = len(dos_list)

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

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                sharex=sharex, sharey=sharey)
        if n_panels == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows * ncols > 1 else [axes]

        for idx, ((omega, rho), title) in enumerate(zip(dos_list, titles)):
            ax = axes[idx]

            omega_np = omega.cpu().numpy() if hasattr(omega, 'cpu') else omega.numpy()
            rho_np = rho.cpu().numpy() if hasattr(rho, 'cpu') else rho.numpy()

            ax.plot(omega_np, rho_np, **kwargs)
            ax.set_xlabel(xlabel, fontsize=DEFAULT_FONTSIZES['labels'])
            ax.set_ylabel(ylabel, fontsize=DEFAULT_FONTSIZES['labels'])
            ax.set_title(title, fontsize=DEFAULT_FONTSIZES['titles'])
            ax.grid(True, alpha=DEFAULT_STYLING['grid_alpha'])

            if energy_range is not None:
                ax.set_xlim(energy_range)

        # Hide unused subplots
        for idx in range(n_panels, len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()

        return tuple(axes[:n_panels])


class ProjectedDOS(DOSCalculator):
    """
    Projected Density of States (PDOS).

    Computes DOS projected onto specific orbitals using eigenvector weights.
    """

    def __init__(self) -> None:
        """Initialize ProjectedDOS."""
        super().__init__()
        self.pdos: Optional[torch.Tensor] = None
        self.orbital_labels: Optional[list[str]] = None

    def from_eigenvalues(
        self,
        E_k: torch.Tensor,
        U: torch.Tensor,
        omega: torch.Tensor,
        eta: float = 0.02,
        orbital_labels: Optional[list[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute projected DOS from eigenvalues and eigenvectors.

        Args:
            E_k: Eigenvalues at each k-point, shape (N_k, N_band)
            U: Eigenvectors at each k-point, shape (N_k, N_band, N_orb)
            omega: Energy grid for DOS, shape (n_omega,)
            eta: Lorentzian broadening width
            orbital_labels: Labels for orbitals

        Returns:
            (omega, rho) tuple - total DOS, PDOS stored in self.pdos
        """
        N_k, N_band, N_orb = U.shape

        # Flatten eigenvalues and reshape eigenvectors
        eps_flat = E_k.flatten()  # (N_k * N_band,)
        psi = U.reshape(N_k * N_band, N_orb)  # (N_k * N_band, N_orb)

        # Eigenvector weights: |ψ_nk(i)|² (probability for each orbital)
        weights = torch.abs(psi) ** 2

        # Create broadcasting grids
        omega_grid = omega[:, None]  # (n_omega, 1)
        eps_grid = eps_flat[None, :]  # (1, N_k * N_band)

        # Lorentzian: (η/π) / [(ω - ε)² + η²]
        lorentzian = (eta / math.pi) / ((omega_grid - eps_grid) ** 2 + eta ** 2)
        # shape: (n_omega, N_k * N_band)

        # Total DOS: (1/N_k) Σ over all states
        rho = torch.sum(lorentzian, dim=1) / N_k

        # Projected DOS for each orbital
        # PDOS_i(ω) = (1/N_k) Σ_{k,n} |ψ_nk(i)|² * L(ω - ε_nk)
        pdos = torch.zeros((len(omega), N_orb), dtype=torch.float64)
        for i in range(N_orb):
            pdos[:, i] = torch.sum(lorentzian * weights[:, i][None, :], dim=1) / N_k

        self.omega = omega
        self.rho = rho
        self.pdos = pdos
        self.eta = eta
        self.orbital_labels = orbital_labels

        return omega, rho

    def get_projected_dos(self) -> torch.Tensor:
        """
        Get projected DOS values.

        Returns:
            PDOS values, shape (n_omega, N_orb)
        """
        if self.pdos is None:
            raise ValueError("No PDOS computed. Call from_eigenvalues() with eigenvectors.")
        return self.pdos

    def plot_projected(
        self,
        ax: Optional["maxes.Axes"] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        ylabel: str = r"PDOS (states/$|t|$)",
        xlabel: str = r"Energy $\omega$ ($|t|$)",
        title: str = "Projected Density of States",
        fontsize: int = 12,
        stacked: bool = True,
        **kwargs,
    ) -> "maxes.Axes":
        """
        Plot projected DOS.

        Args:
            ax: Matplotlib axis
            energy_range: Energy range for plot
            ylabel: Y-axis label
            xlabel: X-axis label
            title: Plot title
            fontsize: Font size
            stacked: Whether to stack the PDOS curves
            **kwargs: Additional arguments

        Returns:
            Matplotlib axis
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        if self.pdos is None:
            raise ValueError("No PDOS computed. Call from_eigenvalues() with eigenvectors.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        omega_np = self.omega.cpu().numpy()
        pdos_np = self.pdos.cpu().numpy()

        n_omega, n_orb = pdos_np.shape

        if self.orbital_labels is None:
            labels = [f"Orbital {i}" for i in range(n_orb)]
        else:
            labels = self.orbital_labels

        if stacked:
            # Stacked plot
            ax.stackplot(omega_np, *[pdos_np[:, i] for i in range(n_orb)],
                        labels=labels, **kwargs)
            ax.legend(fontsize=fontsize - 2)
        else:
            # Overlaid plot
            colors = plt.cm.tab10(torch.linspace(0, 1, n_orb))
            for i in range(n_orb):
                ax.plot(omega_np, pdos_np[:, i], label=labels[i],
                       color=colors[i], **kwargs)
            ax.legend(fontsize=fontsize - 2)

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.grid(True, alpha=0.3)

        if energy_range is not None:
            ax.set_xlim(energy_range)

        return ax
