# Dependency Analysis: condmatTensor vs Reference Libraries

## Executive Summary

This document analyzes the condmatTensor architecture plan and compares it to three established open-source libraries: **NumPy**, **WannierTools**, and **TRIQS**. The goal is to identify architectural patterns and make dependency relations explicit.

---

## 1. Reference Libraries Architecture Analysis

### 1.1 NumPy - The Foundation Layer

**Architecture Pattern:** Layered C/Fortran Core → Python Wrapper

```
┌─────────────────────────────────────────────────────────────┐
│                    NumPy User API                            │
│  numpy.ndarray, numpy.linalg, numpy.fft, numpy.random       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Python C-API Layer                        │
│                  (CPython integration)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 C/Fortran Core Libraries                     │
│        BLAS, LAPACK, FFTW (external dependencies)            │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
| Principle | Description |
|-----------|-------------|
| **Minimal External Dependencies** | Only BLAS/LAPACK/FFTW at C level |
| **Flat Namespace** | Primary API accessible via `import numpy` |
| **Submodule Isolation** | `numpy.linalg`, `numpy.fft` can be imported separately |
| **Progressive Disclosure** | Simple operations in main namespace, advanced in submodules |

**Dependency Graph:**
```
numpy (root)
├── numpy.core          (0 deps - foundational)
│   └── multiarray      (C extension)
├── numpy.linalg        (+ BLAS/LAPACK)
├── numpy.fft           (+ FFTW)
└── numpy.random        (internal PRNG)
```

---

### 1.2 WannierTools - Fortran-Based Topology Package

**Architecture Pattern:** Monolithic Fortran + Input-Driven Interface

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Files (wt.in, etc.)                 │
│              NAMELIST + INPUT_CARDS format                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Fortran Modules                           │
│  ├── surface_mod (surface states)                           │
│  ├── bulk_mod (bulk properties)                             │
│  ├── wannier_mod (Wannier functions)                        │
│  └── tb_mod (tight-binding models)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              External Dependencies                           │
│                 BLAS, LAPACK                                │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
| Principle | Description |
|-----------|-------------|
| **Fortran-Core** | Performance-critical code in Fortran |
| **Input-Driven** | All computation driven by input files |
| **First-Principle Integration** | Designed for VASP, Wien2k outputs |
| **Modular Fortran** | Separate modules for different physics |

**Dependency Graph:**
```
wanniertools (Fortran executable)
├── BLAS/LAPACK    (linear algebra)
├── (no Python runtime deps - optional Python wrapper)
└── First-principle codes (VASP, Wien2k) - for input generation
```

---

### 1.3 TRIQS - C++/Python Hybrid Framework

**Architecture Pattern:** C++ Core Library + Python Bindings (pybind11/Boost.Python)

```
┌─────────────────────────────────────────────────────────────┐
│                    Python User Interface                     │
│           from triqs.gf import GreenFunction                 │
│           from triqs.operators import *                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Python Bindings Layer                        │
│                 (pybind11/Boost.Python)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    C++ Core Libraries                        │
│  ├── triqs::gf (Green's functions)                          │
│  ├── triqs::lattice (lattice structures)                    │
│  ├── triqs::operators (many-body operators)                 │
│  └── triqs::hilbert_space (Hilbert space)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              External Dependencies                           │
│  HDF5, MPI, OpenMP, C++ STL, Boost, FFTW, BLAS, LAPACK      │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
| Principle | Description |
|-----------|-------------|
| **C++ Performance** | Core algorithms in C++ |
| **Python Flexibility** | Scripting and rapid prototyping |
| **Building Blocks** | Modular components for composing workflows |
| **HDF5 Integration** | Standardized data persistence |

**Dependency Graph:**
```
triqs (C++ library + Python bindings)
├── triqs.gf              (0 internal deps - foundational)
├── triqs.lattice         (+ triqs.gf)
├── triqs.operators       (+ triqs.gf, triqs.hilbert_space)
├── triqs.impurity_solver (+ triqs.gf, triqs.operators)
└── External: HDF5, MPI, OpenMP, Boost, FFTW, BLAS/LAPACK
```

---

## 2. condmatTensor Architecture Analysis

### 2.1 Current Implementation Status (as of 2026-02-03)

**Total Implementation**: ~5,900 lines across LEVEL 1-3, partial LEVEL 4, LEVEL 5, and LEVEL 7

```
src/condmatTensor/
├── core/
│   ├── __init__.py
│   ├── base.py (285 lines)             ✅ BaseTensor with semantic labels, orbital metadata, and Fourier transform
│   ├── types.py (145 lines)            ✅ OrbitalMetadata dataclass for structured orbital metadata
│   ├── device.py (87 lines)            ✅ get_device(), is_cuda_available(), get_default_device()
│   ├── math.py                         ❌ NOT IMPLEMENTED
│   └── gpu_utils.py                    ❌ NOT IMPLEMENTED
│
├── lattice/
│   ├── __init__.py
│   ├── model.py (421 lines)            ✅ BravaisLattice, HoppingModel
│   └── bzone.py (94 lines)             ✅ generate_kmesh, generate_k_path, k_frac_to_cart
│
├── solvers/
│   ├── __init__.py
│   └── diag.py (36 lines)              ✅ diagonalize() function
│   └── ed_cnn.py                       ❌ NOT IMPLEMENTED (uses manybody.ed)
│
├── analysis/
│   ├── __init__.py
│   ├── dos.py (601 lines)              ✅ DOSCalculator, ProjectedDOS with enhanced plotting
│   ├── bandstr.py (626 lines)          ✅ BandStructure with enhanced plotting methods
│   ├── plotting_style.py (87 lines)    ✅ Standardized plotting style constants
│   ├── qgt.py                          ❌ NOT IMPLEMENTED
│   ├── topology.py                     ❌ NOT IMPLEMENTED
│   └── fermi.py                        ❌ NOT IMPLEMENTED
│
├── manybody/                           ✅ PARTIAL (preprocessing, magnetic)
│   ├── __init__.py                     ✅ Exports implemented classes
│   ├── preprocessing.py (496 lines)    ✅ Matsubara frequencies, BareGreensFunction, SelfEnergy, SpectralFunction
│   ├── magnetic.py (840 lines)         ✅ LocalMagneticModel, KondoLatticeSolver, SpinFermionModel, pauli_matrices
│   ├── dmft.py                         ❌ NOT IMPLEMENTED (DMFTLoop)
│   ├── cdmft.py                        ❌ NOT IMPLEMENTED (ClusterDMFTLoop)
│   ├── ipt.py                          ❌ NOT IMPLEMENTED (IPTSolver)
│   └── ed.py                           ❌ NOT IMPLEMENTED (ED)
│
├── transport/                          ❌ ENTIRE MODULE NOT IMPLEMENTED
│   ├── __init__.py
│   ├── rgf.py
│   └── transport.py
│
├── optimization/                       ✅ PARTIAL (bayesian, magnetic)
│   ├── __init__.py                     ✅ Exports implemented classes
│   ├── bayesian/                       # Bayesian optimization with multiple backends
│   │   ├── __init__.py (466 lines)     ✅ BayesianOptimizer, MultiObjectiveOptimizer
│   │   ├── sober_backend.py (365 lines) ✅ SOBER backend implementation
│   │   ├── botorch_backend.py (280 lines) ✅ BoTorch backend implementation
│   │   ├── simple_backend.py (239 lines)  ✅ Simple backend (Thompson sampling)
│   │   └── utils.py (128 lines)        ✅ Latin Hypercube Sampling, utilities
│   ├── magnetic.py (631 lines)         ✅ EffectiveArrayOptimizer for Kondo/spin-fermion model downfolding
│   └── ml_interface.py                 ❌ NOT IMPLEMENTED
│
├── interface/                          ❌ ENTIRE MODULE NOT IMPLEMENTED
│   ├── __init__.py
│   ├── yaml_reader.py
│   └── wannier_reader.py
│
└── logging/                            ❌ NOT IMPLEMENTED
    └── __init__.py
```

### 2.2 Dependency Graph (Current Implementation vs Planned)

```
condmatTensor
│
├── Layer 1: Foundation (0 internal deps)
│   └── core/
│       ├── base.py         ✅ (BaseTensor class - 285 lines)
│       ├── types.py        ✅ (OrbitalMetadata dataclass - 145 lines)
│       ├── device.py       ✅ (Device management - 87 lines)
│       ├── math.py         ❌ NOT IMPLEMENTED
│       └── gpu_utils.py    ❌ NOT IMPLEMENTED
│
├── Layer 2: Data Structures (+ core)
│   └── lattice/
│       ├── model.py        ✅ (BravaisLattice, HoppingModel - 421 lines)
│       └── bzone.py        ✅ (generate_kmesh, generate_k_path, k_frac_to_cart - 94 lines)
│
├── Layer 3: Solvers (+ core, lattice)
│   ├── diag.py            ✅ (diagonalize - 36 lines)
│   └── ed_cnn.py          ❌ NOT IMPLEMENTED (uses manybody.ed)
│
├── Layer 4: Many-Body (+ core, lattice)
│   ├── preprocessing.py   ✅ (Matsubara frequencies, BareGreensFunction, SelfEnergy, SpectralFunction - 496 lines)
│   ├── magnetic.py        ✅ (LocalMagneticModel, KondoLatticeSolver, SpinFermionModel, pauli_matrices - 840 lines)
│   ├── dmft.py            ❌ NOT IMPLEMENTED (DMFTLoop)
│   ├── cdmft.py           ❌ NOT IMPLEMENTED (ClusterDMFTLoop)
│   ├── ipt.py             ❌ NOT IMPLEMENTED (IPTSolver)
│   └── ed.py              ❌ NOT IMPLEMENTED (ED)
│
├── Layer 5: Analysis (+ core, lattice, solvers)
│   ├── dos.py             ✅ (DOSCalculator, ProjectedDOS - 601 lines)
│   ├── bandstr.py         ✅ (BandStructure with enhanced plotting - 626 lines)
│   ├── plotting_style.py  ✅ (Plotting style constants - 87 lines)
│   ├── fermi.py           ❌ NOT IMPLEMENTED
│   ├── qgt.py             ❌ NOT IMPLEMENTED
│   └── topology.py        ❌ NOT IMPLEMENTED
│
├── Layer 6: Transport (+ core, lattice)
│   ├── rgf.py             ❌ NOT IMPLEMENTED
│   └── transport.py       ❌ NOT IMPLEMENTED
│
├── Layer 7: Optimization (+ manybody)
│   ├── bayesian/
│   │   ├── __init__.py     ✅ (BayesianOptimizer, MultiObjectiveOptimizer - 466 lines)
│   │   ├── sober_backend.py ✅ (SOBER backend - 365 lines)
│   │   ├── botorch_backend.py ✅ (BoTorch backend - 280 lines)
│   │   ├── simple_backend.py ✅ (Simple backend - 239 lines)
│   │   └── utils.py        ✅ (Utilities - 128 lines)
│   ├── magnetic.py        ✅ (EffectiveArrayOptimizer - 631 lines)
│   └── ml_interface.py    ❌ NOT IMPLEMENTED
│
├── Layer 8: Interface (+ core, lattice)
│   ├── yaml_reader.py     ❌ NOT IMPLEMENTED
│   └── wannier_reader.py  ❌ NOT IMPLEMENTED
│
└── Layer 9: Logging (independent)
    └── __init__.py        ❌ NOT IMPLEMENTED
```

### 2.3 External Dependencies (Current vs Planned)

```
condmatTensor
├── torch>=2.0              ✅ ACTIVELY USED (tensor operations, autograd, GPU)
├── numpy>=1.24             ✅ ACTIVELY USED (numerical utilities)
├── matplotlib>=3.7         ✅ ACTIVELY USED (plotting in analysis modules)
├── scipy>=1.10             ⚠️ IN REQUIREMENTS BUT NOT YET USED (planned for future modules)
├── pyyaml>=6.0             ⚠️ IN REQUIREMENTS BUT NOT YET USED (planned for interface module)
└── scikit-optimize>=0.9    ❌ NOT IN REQUIREMENTS (planned for optimization module)
└── spglib>=2.0             ❌ NOT IN REQUIREMENTS (planned for LEVEL 10 symmetry)
```

---

## 2.4 Current API Reference (Implemented Components)

This section documents all currently implemented classes and their methods.

### LEVEL 1: Core (`src/condmatTensor/core/`)

```python
class BaseTensor:
    """Unified tensor class for all physics objects (Hamiltonian, Green's functions, etc.)

    Attributes:
        tensor: torch.Tensor           # Data
        labels: List[str]              # Semantic labels (e.g., ['k', 'orb_i', 'orb_j'])
        orbital_names: List[str] | None  # Physical orbital names
        orbital_metadatas: List[OrbitalMetadata] | None  # Structured orbital metadata
        displacements: torch.Tensor | None  # For R→k Fourier transforms, shape (N_R, dim)
    """

    def __init__(self, tensor, labels, orbital_names=None, orbital_metadatas=None, displacements=None):
        """Initialize BaseTensor with validation."""

    def to_k_space(self, k) -> 'BaseTensor':
        """Fourier transform H(R) → H(k): H(k) = Σ_R H(R)·exp(i·k·R)"""

    def to(self, device) -> 'BaseTensor':
        """Move tensor to device (CPU/GPU)."""

    # Orbital metadata methods
    def get_f_orbitals(self) -> List[int]:
        """Get indices of f-orbitals."""

    def get_orbitals_by_site(self, site: str) -> List[int]:
        """Get indices of orbitals at a specific site."""

    def get_spinful_orbitals(self) -> List[int]:
        """Get indices of spinful orbitals."""

    def is_spinful_system(self) -> bool:
        """Check if system is spinful."""

    def get_localized_orbitals(self) -> List[int]:
        """Get indices of localized orbitals."""

    @property
    def shape(self) -> torch.Size:
        """Return tensor shape."""

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""

    @property
    def dtype(self) -> torch.dtype:
        """Return tensor dtype."""
```

```python
class OrbitalMetadata:
    """Structured metadata for a single orbital.

    Attributes:
        site: str | None              # Site identifier (e.g., 'Ce1', 'atom1')
        orb: str | None               # Orbital type (e.g., 's', 'px', 'dxy', 'f')
        spin: str | None              # Spin projection ('up', 'down', or None)
        local: bool | None            # Localized (True) vs conductive (False)
        U: float | None               # Hubbard U parameter
        name: str | None              # Display name override
    """

    def to_string(self) -> str:
        """Convert to string format (e.g., 'Ce1-f-spin_up-local-U7.0')."""

    def is_f_orbital(self) -> bool:
        """Check if this is an f-orbital."""

    def is_spinful(self) -> bool:
        """Check if this orbital has spin information."""

    def is_localized(self) -> bool:
        """Check if this orbital is localized."""

    @classmethod
    def from_string(cls, s: str) -> 'OrbitalMetadata':
        """Parse orbital metadata from string."""

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""

    @classmethod
    def from_dict(cls, d: dict) -> 'OrbitalMetadata':
        """Create from dictionary."""
```

```python
def get_device(device=None) -> torch.device:
    """Get device with automatic CUDA detection and CPU fallback.

    Args:
        device: str or torch.device, or None for auto-detection

    Returns:
        torch.device: 'cuda' if available, else 'cpu'
    """

def is_cuda_available() -> bool:
    """Check if CUDA is available on the system."""

def get_default_device() -> torch.device:
    """Returns default CPU device for the library."""
```

### LEVEL 2: Lattice (`src/condmatTensor/lattice/`)

```python
class BravaisLattice:
    """Bravais lattice with multiple sites per unit cell.

    Attributes:
        cell_vectors: torch.Tensor       # Lattice vectors, shape (dim, dim)
        basis_positions: List[torch.Tensor]  # Basis positions (fractional)
        num_orbitals: int                # Orbitals per site
        dim: int                         # Spatial dimension (2 or 3)
    """

    def __init__(self, cell_vectors, basis_positions, num_orbitals=1):
        """Initialize BravaisLattice."""

    @property
    def num_basis(self) -> int:
        """Number of basis sites."""

    @property
    def total_orbitals(self) -> int:
        """Total orbitals (num_sites × num_orbitals)."""

    def reciprocal_vectors(self) -> torch.Tensor:
        """Compute reciprocal lattice vectors."""

    def high_symmetry_points(self) -> dict[str, torch.Tensor]:
        """Return high-symmetry points (G, K, M for triangular)."""

class HoppingModel:
    """General tight-binding model builder with symbolic hopping terms.

    Attributes:
        lattice: BravaisLattice
        hoppings: List[Tuple]  # (orb_i, orb_j, displacement, value)
        orbital_labels: List[str]
    """

    def __init__(self, lattice, orbital_labels=None, hoppings=None):
        """Initialize model."""

    def add_hopping(self, orb_i, orb_j, displacement, value=1.0, add_hermitian=True):
        """Add hopping term. Supports indices or string labels."""

    def build_HR(self) -> BaseTensor:
        """Build real-space Hamiltonian H(R)."""

    def build_Hk(self, k_path) -> BaseTensor:
        """Build k-space Hamiltonian H(k) directly from hoppings."""

def generate_kmesh(lattice, nk, device=None) -> torch.Tensor:
    """Generate uniform k-mesh, returns shape (nk^dim, dim)."""

def generate_k_path(lattice, points, n_per_segment, device=None) -> tuple:
    """Generate k-path along high-symmetry lines. Returns (k_path, ticks)."""

def k_frac_to_cart(k_frac, lattice) -> torch.Tensor:
    """Convert k-points from fractional to Cartesian coordinates."""
```

### LEVEL 3: Solvers (`src/condmatTensor/solvers/diag.py`)

```python
def diagonalize(Hk: torch.Tensor, hermitian: bool = True) -> tuple:
    """Diagonalize Hamiltonian at each k-point.

    Args:
        Hk: Hamiltonian, shape (N_k, N_orb, N_orb)
        hermitian: Use eigh if True (faster), eig if False

    Returns:
        eigenvalues: shape (N_k, N_orb)
        eigenvectors: shape (N_k, N_orb, N_orb)
    """
```

### LEVEL 4: Many-Body (`src/condmatTensor/manybody/`)

```python
def generate_matsubara_frequencies(beta, n_max, fermionic=True, device=None) -> torch.Tensor:
    """Generate Matsubara frequency grid.
    Fermionic: iωₙ = iπ(2n + 1)/β, Bosonic: iΩₙ = i2πn/β

    Returns: Complex tensor of frequencies, shape (2*n_max + 1,)
    """

class BareGreensFunction:
    """Bare (non-interacting) Green's function G₀(iωₙ).

    Attributes:
        omega: torch.Tensor | None
        G0: torch.Tensor | None  # shape (n_omega, N_orb, N_orb)
    """

    def from_hamiltonian(self, Hk, omega, mu=0.0, eta=1e-6):
        """Compute G₀(iωₙ) = (iωₙ + μ + η·sgn(Imω) - Hₖ)⁻¹"""

class SelfEnergy:
    """Self-energy Σ object for DMFT and other many-body methods.

    Attributes:
        omega: torch.Tensor | None
        sigma: torch.Tensor | None  # shape (n_omega, N_orb, N_orb)
    """

    def initialize_atomic(self, U, interaction='half_filling'):
        """Initialize Σ from atomic limit."""

    def from_greens_function(self, G, G0):
        """Compute Σ = G₀⁻¹ - G⁻¹ via Dyson equation."""

class SpectralFunction:
    """Spectral function A(ω) from retarded Green's function.

    Attributes:
        omega: torch.Tensor | None
        spectral: torch.Tensor | None  # shape (n_omega, N_k, N_orb)
    """

    def from_self_energy(self, Hk, omega, sigma, eta=1e-3):
        """Compute A(ω) = (-1/π) Im Gᵣ(ω) with Σ."""

def pauli_matrices(device=None) -> dict[str, torch.Tensor]:
    """Return Pauli matrices {'x', 'y', 'z', 'I'} as 2×2 tensors."""

class LocalMagneticModel:
    """Local magnetic moments coupled to conduction electrons.

    Attributes:
        S: torch.Tensor  # Local spins, shape (N_sites, 3)
        J: float         # Exchange coupling
    """

    def compute_exchange_field(self, S_converged=None):
        """Compute effective Zeeman field from local moments."""

class KondoLatticeSolver:
    """Kondo lattice model solver (H = H₀ + J Σᵢ Sᵢ·sᵢ).

    Attributes:
        H0: BaseTensor
        J: float
        S_positions: list
        mixing: float
    """

    def solve(self, max_iter=1000, tol=1e-10):
        """Self-consistent Kondo lattice calculation."""

class SpinFermionModel:
    """Spin-fermion model with classical spin field.

    Attributes:
        H0: BaseTensor
        J: float
        q: torch.Tensor  # Ordering wavevector
    """

    def build_hamiltonian(self, S_field):
        """Build H with spin-dependent hopping."""
```

### LEVEL 7: Optimization (`src/condmatTensor/optimization/`)

```python
class BayesianOptimizer:
    """Bayesian optimization with multiple backends (SOBER, BoTorch, Simple).

    Attributes:
        bounds: List[Tuple[float, float]]  # Parameter bounds
        backend: str  # 'sober', 'botorch', 'simple', or 'auto'
        n_init: int  # Number of initial random points
        n_iter: int  # Number of optimization iterations
    """

    def optimize(self, objective, maximize=False, verbose=True, device=None) -> Tuple:
        """Run optimization. Returns (X_best, y_best)."""

    def get_best(self) -> Tuple:
        """Get best observation (X, y)."""

    def reset(self) -> None:
        """Reset optimizer state."""

class MultiObjectiveOptimizer:
    """Multi-objective Bayesian optimization (Pareto front).

    Attributes:
        bounds: List[Tuple[float, float]]
        n_objectives: int
        n_init: int
        n_iter: int
    """

    def optimize(self, objective, verbose=True, device=None) -> Tuple:
        """Run multi-objective optimization."""

    def _get_pareto_front(self) -> List:
        """Extract Pareto-optimal solutions."""

class SoberBackend:
    """SOBER (Sequential Optimization using Ensemble of Regressors) backend.

    Reference: https://github.com/ma921/SOBER
    """

class BotorchBackend:
    """BoTorch Gaussian Process backend.

    Reference: https://botorch.org/
    """

class SimpleBackend:
    """Fallback backend using Thompson sampling or random search."""

def latin_hypercube_sampling(bounds, n_samples, device, seed=None) -> torch.Tensor:
    """Latin Hypercube Sampling for initial Bayesian optimization points."""

class EffectiveArrayOptimizer:
    """Effective model downfolding for Kondo/spin-fermion models.

    Attributes:
        full_model: SpinFermionModel | KondoLatticeSolver
        target_orbitals: list
    """

    def optimize_effective_model(self, omega_range, **opt_kwargs):
        """Find optimal hopping parameters for effective model."""
```

### LEVEL 5: Analysis (`src/condmatTensor/analysis/`)

```python
# Plotting style constants for publication-quality plots
DEFAULT_FIGURE_SIZES = {
    'single': (6, 5),
    'dual': (14, 5),
    'triple': (18, 5),
    '2x2': (12, 10),
    'band_dos': (10, 5),
    'comparison_2x3': (15, 10),
    '2x1_vertical': (6, 10),
    '3x1': (12, 10),
    'wide': (16, 5),
}
DEFAULT_COLORS = {...}  # Color scheme (primary, reference, secondary, etc.)
DEFAULT_FONTSIZES = {...}  # Font sizes for labels, titles, legend
DEFAULT_STYLING = {...}  # Styling options (grid_alpha, band_linewidth, dpi)
DEFAULT_COLORMAPS = {...}  # Colormaps (orbital_weight, spin, orbitals)
LINE_STYLES = {...}  # Line styles (solid, dashed, dotted, dash_dot)
MARKER_STYLES = {...}  # Marker styles (circle, square, triangle, etc.)

class DOSCalculator:
    """Density of States calculator with Lorentzian broadening.

    Attributes:
        omega: torch.Tensor | None
        rho: torch.Tensor | None
        eta: float | None
    """

    def __init__(self):
        """Initialize calculator."""

    def from_eigenvalues(self, E_k, omega, eta=0.02):
        """Compute DOS: ρ(ω) = (1/N_k) Σ (η/π) / [(ω-ε)² + η²]"""

    def plot(self, ax=None, energy_range=None, ylabel, xlabel, title, fontsize, fill, **kwargs):
        """Plot stored DOS results."""

    def plot_with_reference(self, reference_energies, labels=None, colors=None, ...):
        """Plot DOS with vertical reference lines."""

    def plot_comparison(self, other_dos_data, labels, colors=None, ...):
        """Overlay multiple DOS curves."""

    def plot_multi_panel(self, dos_list, titles, figsize=None, ...):
        """Create multi-panel DOS comparison."""

class ProjectedDOS(DOSCalculator):
    """Projected Density of States (extends DOSCalculator).

    Additional Attributes:
        pdos: torch.Tensor | None  # shape (n_omega, N_orb)
        orbital_labels: list[str] | None
    """

    def __init__(self):
        """Initialize (inherits from DOSCalculator)."""

    def from_eigenvalues(self, E_k, U, omega, eta=0.02, orbital_labels=None):
        """Compute PDOS from eigenvalues and eigenvectors."""

    def get_projected_dos(self) -> torch.Tensor:
        """Return PDOS tensor."""

    def plot_projected(self, ax=None, energy_range=None, ylabel, xlabel, title, fontsize, stacked, **kwargs):
        """Plot PDOS (stacked or overlaid)."""

class BandStructure:
    """Band structure calculator with enhanced plotting methods.

    Attributes:
        k_path: torch.Tensor | None
        eigenvalues: torch.Tensor | None
        ticks: List[Tuple[int, str]] | None
    """

    def __init__(self):
        """Initialize calculator."""

    def compute(self, eigenvalues, k_path, ticks=None):
        """Store band structure results."""

    def plot(self, ax=None, energy_range=None, ylabel, title, fontsize, **kwargs):
        """Plot band structure."""

    def plot_with_dos(self, eigenvalues_mesh, omega, eta=0.02, energy_range, ylabel, dos_xlabel,
                     title, fontsize, figsize, dos_color, **kwargs):
        """Combined plot: bands + DOS (shared y-axis)."""

    def plot_colored_by_weight(self, eigenvectors, orbital_indices, ax=None, cmap='viridis', ...):
        """Plot bands colored by orbital weight."""

    def add_reference_line(self, energy, label=None, color='red', linestyle='--', ...):
        """Add horizontal reference line."""

    def plot_comparison(self, other_eigenvalues, labels, colors=None, ...):
        """Overlay multiple band structures."""

    def plot_multi_panel(self, eigenvalues_list, titles, k_paths=None, ...):
        """Create multi-panel comparison."""
```

---

## 3. Comparative Analysis

### 3.1 Architecture Pattern Comparison

| Aspect | NumPy | WannierTools | TRIQS | condmatTensor |
|--------|-------|--------------|-------|---------------|
| **Core Language** | C/Fortran | Fortran | C++ | Python |
| **Python API** | Native | Optional wrapper | pybind11/Boost | Native |
| **Primary Backend** | BLAS/LAPACK | BLAS/LAPACK | HDF5/MPI | PyTorch |
| **Modularity** | High (submodules) | Medium (Fortran modules) | High (C++ namespaces) | High (planned) |
| **GPU Support** | Via CuPy/NumPy-compat | No | No (CPU) | Yes (PyTorch native) |
| **Extensibility** | C-API | Fortran source | C++ inheritance | Python composition |

### 3.2 Dependency Management Comparison

| Library | Internal Dep Strategy | External Dep Strategy |
|---------|----------------------|----------------------|
| **NumPy** | Layered: core → submodules | Minimal: BLAS/LAPACK only |
| **WannierTools** | Monolithic Fortran | BLAS/LAPACK + first-principle codes |
| **TRIQS** | Building blocks pattern | Heavy: HDF5, MPI, Boost, etc. |
| **condmatTensor** | **Layered (planned)** | **Medium: PyTorch ecosystem** |

### 3.3 Key Differences

#### 3.3.1 NumPy vs condmatTensor
- **NumPy**: C-optimized array operations, minimal external deps
- **condmatTensor**: PyTorch-based (autograd, GPU native), domain-specific

#### 3.3.2 WannierTools vs condmatTensor
- **WannierTools**: Fortran monolith, input-file driven
- **condmatTensor**: Python native, programmatic API, no input files

#### 3.3.3 TRIQS vs condmatTensor
- **TRIQS**: C++ core with Python bindings, HDF5 persistence
- **condmatTensor**: Pure Python, PyTorch tensors (in-memory)

---

## 4. Recommendations for Clearer Dependencies

### 4.1 Make Internal Dependencies Explicit

Create a **`DEPENDENCIES.md`** file at project root:

```markdown
# condmatTensor Internal Dependencies

## Dependency Levels

### Level 0: External Dependencies (No internal deps)
- torch>=2.0
- numpy>=1.24
- matplotlib>=3.7
- scipy>=1.10
- pyyaml>=6.0
- scikit-optimize>=0.9

### Level 1: Core Modules (No internal deps)
- `condmatTensor.core.base`
- `condmatTensor.core.math`
- `condmatTensor.core.gpu_utils`
- `condmatTensor.logging`

### Level 2: Data Structures (+ Level 1)
- `condmatTensor.lattice.model`
- `condmatTensor.lattice.bzone`

### Level 3: Solvers (+ Level 1, Level 2)
- `condmatTensor.solvers.diag`
- `condmatTensor.solvers.ed_cnn` (+ Level 4: manybody.ed)

### Level 4: Many-Body (+ Level 1, Level 2)
- `condmatTensor.manybody.preprocessing`
- `condmatTensor.manybody.dmft` (+ manybody.ipt)
- `condmatTensor.manybody.cdmft` (+ manybody: ipt, ed)
- `condmatTensor.manybody.ipt`
- `condmatTensor.manybody.ed`

### Level 5: Analysis (+ Level 1, Level 2, Level 3)
- `condmatTensor.analysis.dos`
- `condmatTensor.analysis.bandstr`
- `condmatTensor.analysis.fermi`
- `condmatTensor.analysis.qgt` (+ core.math)
- `condmatTensor.analysis.topology` (+ analysis.qgt)

### Level 6: Transport (+ Level 1, Level 2)
- `condmatTensor.transport.rgf`
- `condmatTensor.transport.transport` (+ transport.rgf)

### Level 7: Optimization (+ Level 4: manybody)
- `condmatTensor.optimization.bayesian`
- `condmatTensor.optimization.ml_interface`

### Level 8: Interface (+ Level 1, Level 2)
- `condmatTensor.interface.yaml_reader`
- `condmatTensor.interface.wannier_reader`
```

### 4.2 Update `__init__.py` Files for Dependency Control

Each `__init__.py` should explicitly control what's exported:

```python
# src/condmatTensor/__init__.py
"""
condmatTensor: PyTorch-based condensed matter physics library.

Dependency Levels:
- Level 1: core (foundation)
- Level 2: lattice (data structures)
- Level 3: solvers (computational engines)
- Level 4: manybody (many-body physics)
- Level 5: analysis (observables)
- Level 6: transport (transport properties)
- Level 7: optimization (parameter search)
- Level 8: interface (external I/O)
"""

# Level 1: Core (always safe to import)
from .core import BaseTensor, berry_curvature, chern_number

# Level 2: Lattice (requires only core)
from .lattice import BravaisLattice, ClusterLattice, generate_kmesh, generate_k_path

# Level 3+: Import on demand to avoid circular imports
# Use explicit imports: from condmatTensor.solvers import diagonalize
# from condmatTensor.manybody import DMFTLoop, ClusterDMFTLoop
# etc.

__all__ = [
    # Level 1
    'BaseTensor', 'berry_curvature', 'chern_number',
    # Level 2
    'BravaisLattice', 'ClusterLattice', 'generate_kmesh', 'generate_k_path',
]
```

### 4.3 Add Dependency Graph Visualization

Create a `docs/dependencies.svg` showing module relationships.

---

## 5. Architectural Improvements Based on Reference Libraries

### 5.1 From NumPy: Progressive Disclosure Pattern

```python
# Main namespace: common operations
import condmatTensor as cmt

# Submodules: specialized functionality
from condmatTensor.analysis import topology
from condmatTensor.manybody import DMFTLoop, ClusterDMFTLoop
```

### 5.2 From TRIQS: Building Blocks Pattern

```python
# Composable components for workflows
Hk = cmt.BaseTensor(...)          # Building block 1
lattice = cmt.BravaisLattice(...)  # Building block 2
dmft = cmt.manybody.DMFTLoop(Hk, omega, solver)  # Composed
```

### 5.3 From WannierTools: Clear Input/Output Contract

```python
# Although not file-driven, define clear input/output contracts
def build_kagome_lattice(a=1.0, t=-1.0) -> tuple[BravaisLattice, BaseTensor]:
    """Returns: (lattice, Hr) with explicit types."""
```

---

## 6. Dependency Safety Rules

Based on the reference libraries analysis:

1. **No Circular Imports**: Each level depends only on lower levels
2. **Explicit Imports**: Always use `from condmatTensor.X import Y` not relative imports
3. **Lazy Loading**: Heavy modules (manybody, optimization) imported on demand
4. **Type Annotations**: All functions have explicit type hints
5. **Testing**: Each level has independent tests

---

## 7. Updated Dependency Diagram (Visual)

```
                        ┌─────────────────────────────────────┐
                        │      condmatTensor (root)           │
                        └─────────────────────────────────────┘
                                                 │
          ┌──────────────────────────────────────┼──────────────────────────────────────┐
          │                                      │                                      │
          ▼                                      ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐              ┌─────────────────────┐
│   External Deps     │              │  Level 1: Core      │              │  Level 9: Logging    │
│  • torch>=2.0       │              │  • base.py          │              │  (independent)      │
│  • numpy>=1.24      │              │  • math.py          │              │                     │
│  • matplotlib>=3.7  │              │  • gpu_utils.py     │              │                     │
│  • scipy>=1.10      │              └─────────────────────┘              └─────────────────────┘
│  • pyyaml>=6.0      │                        │
│  • scikit-optimize  │                        │
└─────────────────────┘                        ▼
                              ┌─────────────────────────────────────┐
                              │       Level 2: Lattice               │
                              │  • model.py (BravaisLattice)         │
                              │  • bzone.py (k-mesh, paths)          │
                              └─────────────────────────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
    ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
    │  Level 3: Solvers   │      │  Level 8: Interface  │      │  Level 5: Analysis  │
    │  • diag.py          │      │  • yaml_reader.py   │      │  • dos.py           │
    │  • ipt.py           │      │  • wannier_reader.py│      │  • bandstr.py       │
    │  • ed.py            │      └─────────────────────┘      │  • fermi.py         │
    │  • ed_cnn.py        │                                    │  • qgt.py           │
    └─────────────────────┘                                    │  • topology.py     │
           │                                                   └─────────────────────┘
           │                                                            │
           ▼                                                            │
    ┌─────────────────────┐                                            │
    │  Level 4: DMFT      │                                            │
    │  • preProcessing.py │                                            │
    │  • singlesite.py    │                                            │
    │  • cluster.py       │                                            │
    └─────────────────────┘                                            │
           │                                                            │
           ▼                                                            │
    ┌─────────────────────┐                                            │
    │ Level 7: Optimize   │                                            │
    │  • bayesian.py      │                                            │
    │  • ml_interface.py  │                                            │
    └─────────────────────┘                                            │
                                                                      │
           ┌────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  Level 6: Transport │
    │  • rgf.py           │
    │  • transport.py     │
    └─────────────────────┘
```

---

## 8. Conclusion

The condmatTensor plan follows a **layered architecture** similar to NumPy, with clear separation between foundation (core) and specialized modules (manybody, transport, optimization). Key differences from reference libraries:

1. **Pure Python** (unlike NumPy/TRIQS with compiled cores)
2. **PyTorch-based** (GPU-native autograd, unlike NumPy's CPU-first approach)
3. **Domain-specific** (condensed matter, unlike NumPy's general-purpose focus)

**Recommended next steps:**
1. Add `DEPENDENCIES.md` to project root
2. Update each `__init__.py` with explicit `__all__` exports
3. Add type annotations to all public APIs
4. Create visual dependency graph in documentation
