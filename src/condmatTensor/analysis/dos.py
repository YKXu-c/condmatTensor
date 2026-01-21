"""Density of States (DOS) calculator with Lorentzian broadening.

Future Work:
    - from_green_function(): Compute DOS from spectral function A(ω) = -(1/π) Im[Gᴿ(ω)]
    - from_spectral_function(): Direct A(ω) input with k-resolved averaging
"""

from typing import Optional, Tuple
import torch
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.axes as maxes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


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
