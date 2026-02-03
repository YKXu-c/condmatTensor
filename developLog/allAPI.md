# condmatTensor - Complete API Documentation

**Version**: 0.0.1
**License**: MIT
**Implementation Status**: ~50% complete (5 of 10 levels partially/fully implemented, ~7,500 lines)
**Recent Updates (2026-02-03)**:
- DMFT loop + IPT impurity solver with ABC architecture
- **CRITICAL BUG FIXES**: IPT now correctly uses G₀ (Weiss field) instead of G_loc, added 1/β FFT normalization, fermionic boundary condition enforcement
- Pade analytic continuation implementation with modular class-based framework
- Auto-calculated DOS range for interacting systems
- DMFT example validation with 11 plots (including Im[G(iωₙ)] for TRIQS-style validation, U-scan 3x3 grid, and vertical DOS stack)
- **t_f parameter scan**: f-f hopping/f-d hybridization scan (-0.1 to -1.0) with organized output directories
- **Pade simple test example**: Standalone test for analytic continuation without DMFT (5 test cases, 5 plots)

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Key Conventions](#key-conventions)
- [LEVEL 1: Core](#level-1-core)
- [LEVEL 2: Lattice](#level-2-lattice)
- [LEVEL 3: Solvers](#level-3-solvers)
- [LEVEL 4: Many-Body](#level-4-many-body)
- [LEVEL 5: Analysis](#level-5-analysis)
- [LEVEL 7: Optimization](#level-7-optimization)
- [Import Reference](#import-reference)

---

## Overview

**condmatTensor** is a PyTorch-based condensed matter physics library for quantum materials research. The library uses a **unified tensor-first approach** where all physics objects (Hamiltonians, Green's functions, self-energies) are represented by a single `BaseTensor` class with semantic labels.

### Key Features

- **One Class for All Physics Objects**: `BaseTensor` handles H, G, and Σ with automatic R→k Fourier transforms
- **GPU Acceleration**: Native PyTorch CUDA support with automatic CPU fallback
- **Spinor Convention**: Magnetic systems use embedded spin in orbital indices
- **Dimensionless Units**: All energies in units of hopping parameter |t|

### Module Structure

```
condmatTensor/
├── core/           (LEVEL 1) ✅ Complete
├── lattice/        (LEVEL 2) ✅ Complete
├── solvers/        (LEVEL 3) ⚠️ Partial
├── manybody/       (LEVEL 4) ⚠️ Partial
├── analysis/       (LEVEL 5) ⚠️ Partial
├── transport/      (LEVEL 6) ❌ Not started
├── optimization/   (LEVEL 7) ⚠️ Partial
├── interface/      (LEVEL 8) ❌ Not started
├── logging/        (LEVEL 9) ❌ Not started
└── symmetry/       (LEVEL 10) ❌ Not started
```

---

## Installation

### Critical Installation Order

**PyTorch MUST be installed FIRST via CUDA-specific URL:**

```bash
# Step 1: Install PyTorch 2.10+ with CUDA 13.0 support FIRST
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Step 2: Install remaining dependencies
pip install -r requirements.txt

# Step 3: Set PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

### Dependencies

| Package | Version | Required |
|---------|---------|----------|
| `torch` | >=2.10 | Yes |
| `numpy` | >=1.24, <2.0 | Yes |
| `matplotlib` | >=3.7 | Yes |
| `scipy` | >=1.10 | Yes |
| `pyyaml` | >=6.0 | Yes |
| `sober-bo` | ==2.0.4 | Optional (LEVEL 7) |
| `botorch` | >=0.9.0 | Optional (LEVEL 7) |

---

## Key Conventions

### 1. BaseTensor - Unified Representation

All physics objects use semantic labels:

```python
from condmatTensor.core import BaseTensor

# Hamiltonian in real space
HR = BaseTensor(tensor, labels=['R', 'orb_i', 'orb_j'], displacements=...)

# Hamiltonian in k-space (auto Fourier transform)
Hk = HR.to_k_space(k_points)  # Returns BaseTensor with labels=['k', 'orb_i', 'orb_j']

# Green's function, self-energy also use BaseTensor
G = BaseTensor(tensor, labels=['k', 'orb_i', 'orb_j', 'iwn'])
```

### 2. Spinor Convention for Magnetic Systems

Spin is embedded in orbital indices: `[orb_0_up, orb_0_down, orb_1_up, orb_1_down, ...]`

```python
# 3 orbital sites x 2 spins = 6 orbitals
num_orbitals = [2, 2, 2]  # Each site has 2 spin states
```

### 3. Coordinate Conventions

| Aspect | Convention |
|--------|------------|
| **User input** | Fractional (units of lattice vectors) |
| **Internal storage** | Cartesian |
| `cell_vectors` | Cartesian |
| `displacements` in `add_hopping()` | Fractional |
| `reciprocal_vectors()` | Cartesian |

### 4. Energy Units

All energies are **dimensionless** in units of hopping parameter `|t|`. Default: `t = -1.0`.

- Kagome eigenvalues: -2|t| to +4|t|
- Flat band at E = -2|t|

### 5. CPU vs GPU Split

| **GPU (Compute-Intensive)** | **CPU (Control Logic & I/O)** |
|---------------------------|------------------------------|
| Bayesian optimization (LEVEL 7) | DMFT loop control (LEVEL 4) |
| Large matrix diagonalization (LEVEL 3) | Plotting with matplotlib |
| Dense tensor operations (all levels) | File I/O (all levels) |
| Lorentzian broadening (LEVEL 5) | Loop overhead (all levels) |

---

## LEVEL 1: Core

**Status**: ✅ Complete
**Path**: `src/condmatTensor/core/`
**Files**: `base.py` (285 lines), `types.py` (145 lines), `device.py` (87 lines)

### Classes

#### `BaseTensor`

Unified tensor class for all physics objects (Hamiltonians, Green's functions, Self-energies).

**File**: `core/base.py:7-139`

```python
BaseTensor(tensor, labels, orbital_names=None, displacements=None)
```

**Parameters**:
- `tensor` (torch.Tensor): Underlying PyTorch tensor
- `labels` (list[str]): Semantic labels for each dimension (e.g., ['R', 'orb_i', 'orb_j'])
- `orbital_names` (list[str], optional): Names for orbitals
- `displacements` (torch.Tensor, optional): Real-space displacement vectors

**Methods**:
- `to_k_space(k_points)` - Fourier transform from R-space to k-space
- `to(device)` - Move tensor to CPU/GPU
- `cpu()` - Move to CPU
- `cuda()` - Move to CUDA
- `numpy()` - Convert to NumPy array

**Properties**:
- `shape` - Tensor shape
- `ndim` - Number of dimensions
- `dtype` - Data type
- `device` - Current device

**Example**:
```python
from condmatTensor.core import BaseTensor
import torch

# Create a Hamiltonian in real space
HR = BaseTensor(
    tensor=torch.randn(10, 6, 6),
    labels=['R', 'orb_i', 'orb_j'],
    displacements=torch.arange(10).reshape(-1, 1)
)

# Fourier transform to k-space
k_points = torch.rand(100, 3)
Hk = HR.to_k_space(k_points)
print(Hk.labels)  # ['k', 'orb_i', 'orb_j']
```

**Orbital Metadata System**:

BaseTensor now supports structured orbital metadata via the `orbital_metadatas` parameter:

```python
from condmatTensor.core import BaseTensor, OrbitalMetadata

# Using OrbitalMetadata objects
orbital_metadatas = [
    OrbitalMetadata(site='Ce1', orb='f', spin='up', local=True, U=7.0),
    OrbitalMetadata(site='Ce1', orb='f', spin='down', local=True, U=7.0),
]

H = BaseTensor(
    tensor=torch.zeros((100, 2, 2)),
    labels=['k', 'orb_i', 'orb_j'],
    orbital_metadatas=orbital_metadatas,
)

# Query orbitals by properties
f_indices = H.get_f_orbitals()  # [0, 1]
is_spinful = H.is_spinful_system()  # True
```

**Methods for Orbital Metadata**:

| Method | Description |
|--------|-------------|
| `get_f_orbitals()` | Get indices of f-orbitals |
| `get_orbitals_by_site(site)` | Get indices of orbitals at a specific site |
| `get_spinful_orbitals()` | Get indices of spinful orbitals |
| `is_spinful_system()` | Check if system is spinful |
| `get_localized_orbitals()` | Get indices of localized orbitals |

---

#### `OrbitalMetadata`

Structured metadata for a single orbital.

**File**: `core/types.py:11-162`

```python
OrbitalMetadata(site=None, orb=None, spin=None, local=None, U=None, name=None)
```

**Parameters**:
- `site` (str, optional): Site identifier (e.g., 'Ce1', 'atom1')
- `orb` (str, optional): Orbital type (e.g., 's', 'px', 'dxy', 'f')
- `spin` (str, optional): Spin projection ('up', 'down', or None)
- `local` (bool, optional): Localized (True) vs conductive (False)
- `U` (float, optional): Hubbard U parameter
- `name` (str, optional): Display name override

**Methods**:

| Method | Description |
|--------|-------------|
| `to_string()` | Convert to string format (e.g., 'Ce1-f-spin_up-local-U7.0') |
| `is_f_orbital()` | Check if this is an f-orbital |
| `is_spinful()` | Check if this orbital has spin information |
| `is_localized()` | Check if this orbital is localized |
| `from_string(s)` | Parse orbital metadata from string (classmethod) |
| `as_dict()` | Convert to dictionary for serialization |
| `from_dict(d)` | Create from dictionary (classmethod) |

**Example**:
```python
from condmatTensor.core import OrbitalMetadata

# Create from parameters
md = OrbitalMetadata(site='Ce1', orb='f', spin='up', local=True, U=7.0)
print(md.to_string())  # 'Ce1-f-spin_up-local-U7.0'

# Parse from string
md2 = OrbitalMetadata.from_string('Ce1-f-spin_up-local-U7.0')
print(md2.site, md2.orb, md2.spin)  # ('Ce1', 'f', 'up')

# Query properties
print(md2.is_f_orbital())  # True
print(md2.is_spinful())  # True
print(md2.is_localized())  # True
```

---

### Functions

#### `get_device()`

Get device with automatic CUDA detection and CPU fallback.

**File**: `core/device.py:15-52`

```python
get_device(device=None)
```

**Parameters**:
- `device` (str or torch.device, optional): Specific device to use, or None for auto-detection

**Returns**:
- `torch.device`: Device object ('cuda' if available, else 'cpu')

**Example**:
```python
from condmatTensor.core import get_device

device = get_device()  # Auto-detect
print(device)  # cuda:0 or cpu
```

---

#### `get_default_device()`

Returns default CPU device for the library.

**File**: `core/device.py:55-70`

```python
get_default_device()
```

**Returns**:
- `torch.device`: CPU device

**Example**:
```python
from condmatTensor.core import get_default_device

cpu_device = get_default_device()
print(cpu_device)  # cpu
```

---

#### `is_cuda_available()`

Check if CUDA is available on the system.

**File**: `core/device.py:73-84`

```python
is_cuda_available()
```

**Returns**:
- `bool`: True if CUDA is available

**Example**:
```python
from condmatTensor.core import is_cuda_available

if is_cuda_available():
    print("GPU acceleration available!")
```

---

## LEVEL 2: Lattice

**Status**: ✅ Complete
**Path**: `src/condmatTensor/lattice/`
**Files**: `model.py` (421 lines), `bzone.py` (94 lines)

### Classes

#### `BravaisLattice`

Represents periodic crystal structure with lattice vectors and basis positions.

**File**: `lattice/model.py:222-375`

```python
BravaisLattice(cell_vectors, basis_positions, num_orbitals)
```

**Parameters**:
- `cell_vectors` (torch.Tensor): Shape (dim, dim) - lattice vectors in Cartesian coordinates
- `basis_positions` (list[tuple[float]]): List of basis atom positions in fractional coords
- `num_orbitals` (list[int]): Number of orbitals per site

**Methods**:

| Method | Description |
|--------|-------------|
| `reciprocal_vectors()` | Compute reciprocal lattice vectors, shape (dim, dim) |
| `orbital_offsets()` | Return cumulative orbital offsets for each site |
| `site_orbital_slice(site_idx)` | Return slice for orbitals at specific site |
| `high_symmetry_points()` | Return high-symmetry points dict (Γ, K, M) |

**Properties**:
- `num_basis` - Number of basis sites
- `total_orbitals` - Total orbitals in unit cell
- `dimension` - Dimension (2 or 3)

**Example**:
```python
from condmatTensor.lattice import BravaisLattice
import torch

# Kagome lattice
cell_vectors = torch.tensor([
    [0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0],
    [0.0, 0.0, 1.0]
])
basis_positions = [(0, 0), (0.5, 0), (0.25, 0.25)]
num_orbitals = [1, 1, 1]  # 1 orbital per site

lattice = BravaisLattice(cell_vectors, basis_positions, num_orbitals)
print(lattice.total_orbitals)  # 3
print(lattice.reciprocal_vectors())  # Reciprocal lattice vectors
```

---

#### `HoppingModel`

Build tight-binding models from symbolic hopping terms.

**File**: `lattice/model.py:7-220`

```python
HoppingModel(lattice, orbital_labels=None, hoppings=None)
```

**Parameters**:
- `lattice` (BravaisLattice): The lattice structure
- `orbital_labels` (list[str], optional): Labels for orbitals
- `hoppings` (list, optional): Initial hopping terms

**Methods**:

| Method | Description |
|--------|-------------|
| `add_hopping(orb_i, orb_j, displacement, value=1.0, add_hermitian=True)` | Add hopping term |
| `build_HR()` | Build real-space Hamiltonian (BaseTensor with labels=['R', 'orb_i', 'orb_j']) |
| `build_Hk(k_path)` | Build k-space Hamiltonian (BaseTensor with labels=['k', 'orb_i', 'orb_j']) |
| `to(device)` | Move model to device |

**Example**:
```python
from condmatTensor.lattice import BravaisLattice, HoppingModel
import torch

# Create Kagome lattice
cell_vectors = torch.tensor([
    [0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0],
    [0.0, 0.0, 1.0]
])
basis_positions = [(0, 0), (0.5, 0), (0.25, 0.25)]
lattice = BravaisLattice(cell_vectors, basis_positions, [1, 1, 1])

# Build tight-binding model
model = HoppingModel(lattice, orbital_labels=['A', 'B', 'C'])
model.add_hopping('A', 'B', [0, 0], 1.0)  # Nearest neighbor
model.add_hopping('B', 'C', [0, 0], 1.0)
model.add_hopping('C', 'A', [0, 0], 1.0)

# Build Hamiltonian in k-space
k_points = torch.rand(100, 2)
Hk = model.build_Hk(k_points)
print(Hk.shape)  # (100, 3, 3) - (N_k, N_orb, N_orb)
```

---

### Functions

#### `generate_kmesh()`

Generate uniform k-mesh in fractional coordinates.

**File**: `lattice/bzone.py:7-32`

```python
generate_kmesh(lattice, nk, device=None)
```

**Parameters**:
- `lattice` (BravaisLattice): The lattice structure
- `nk` (int): Number of k-points per dimension
- `device` (torch.device, optional): Device for output tensor

**Returns**:
- `torch.Tensor`: Shape (nk^dim, dim) - k-points in fractional coordinates

**Example**:
```python
from condmatTensor.lattice import BravaisLattice, generate_kmesh

lattice = BravaisLattice(...)
k_mesh = generate_kmesh(lattice, nk=10)  # 10x10 mesh for 2D
print(k_mesh.shape)  # (100, 2)
```

---

#### `generate_k_path()`

Generate k-point path along high-symmetry lines.

**File**: `lattice/bzone.py:35-77`

```python
generate_k_path(lattice, points, n_per_segment, device=None)
```

**Parameters**:
- `lattice` (BravaisLattice): The lattice structure
- `points` (list[str]): List of high-symmetry point labels (e.g., ['G', 'K', 'M', 'G'])
- `n_per_segment` (int): Number of k-points per segment
- `device` (torch.device, optional): Device for output tensor

**Returns**:
- `tuple`: (k_path, ticks) - k-points and tick positions for labels

**Example**:
```python
from condmatTensor.lattice import BravaisLattice, generate_k_path

lattice = BravaisLattice(...)
k_path, ticks = generate_k_path(lattice, ['G', 'K', 'M', 'G'], n_per_segment=50)
print(k_path.shape)  # (~150, 2)
```

---

#### `k_frac_to_cart()`

Convert k-points from fractional to Cartesian coordinates.

**File**: `lattice/bzone.py:80-95`

```python
k_frac_to_cart(k_frac, lattice)
```

**Parameters**:
- `k_frac` (torch.Tensor): k-points in fractional coordinates
- `lattice` (BravaisLattice): The lattice structure

**Returns**:
- `torch.Tensor`: k-points in Cartesian coordinates

---

## LEVEL 3: Solvers

**Status**: ⚠️ Partial (Only diagonalize implemented, IPT/ED not implemented)
**Path**: `src/condmatTensor/solvers/`
**Files**: `diag.py` (37 lines)

### Functions

#### `diagonalize()`

Diagonalize Hamiltonian at each k-point.

**File**: `solvers/diag.py:7-37`

```python
diagonalize(Hk, hermitian=True)
```

**Parameters**:
- `Hk` (torch.Tensor or BaseTensor): Shape (N_k, N_orb, N_orb)
- `hermitian` (bool): Use eigh if True, eig if False (default: True)

**Returns**:
- `tuple`: (eigenvalues, eigenvectors)
  - `eigenvalues`: torch.Tensor, shape (N_k, N_orb)
  - `eigenvectors`: torch.Tensor, shape (N_k, N_orb, N_orb)

**Example**:
```python
from condmatTensor.lattice import BravaisLattice, HoppingModel, generate_k_path
from condmatTensor.solvers import diagonalize

# Build model
lattice = BravaisLattice(...)
model = HoppingModel(lattice)
model.add_hopping('A', 'B', [0, 0], 1.0)

# Generate k-path and diagonalize
k_path, ticks = generate_k_path(lattice, ['G', 'K', 'M', 'G'], n_per_segment=50)
Hk = model.build_Hk(k_path)

evals, evecs = diagonalize(Hk.tensor)
print(evals.shape)  # (150, 3) - (N_k, N_orb)
```

---

## LEVEL 4: Many-Body

**Status**: ⚠️ Partial (DMFT loop + IPT implemented, ED/NRG/CTQMC pending, analytic continuation framework added)
**Path**: `src/condmatTensor/manybody/`
**Files**: `preprocessing.py` (627 lines), `analytic_continuation.py` (430 lines), `magnetic.py` (840 lines), `impSolvers/base.py` (90 lines), `impSolvers/ipt.py` (310 lines), `dmft.py` (416 lines)

### Functions

#### `generate_matsubara_frequencies()`

Generate Matsubara frequency grid.

**File**: `manybody/preprocessing.py:17-85`

```python
generate_matsubara_frequencies(beta, n_max, fermionic=True, device=None)
```

**Parameters**:
- `beta` (float): Inverse temperature (1/T)
- `n_max` (int): Maximum frequency index
- `fermionic` (bool): Fermionic frequencies if True (default), bosonic if False
- `device` (torch.device, optional): Device for output tensor

**Returns**:
- `torch.Tensor`: Shape (2*n_max + 1,) - Matsubara frequencies

**Indexing Scheme**:
- Frequencies indexed from n = -n_max to n = +n_max (NOT from 0!)
- For n_max=2: indices are [-2, -1, 0, 1, 2]
- The "zero" index (n=0) does NOT mean zero frequency (iω₀ = iπ/β)
- For plotting, use `torch.arange(len(omega))` for x-axis labels

**Formula**:
- Fermionic: iωₙ = iπ(2n + 1)/β
- Bosonic: iωₙ = iπ(2n)/β

**Example**:
```python
from condmatTensor.manybody.preprocessing import generate_matsubara_frequencies

beta = 10.0  # T = 0.1
n_max = 100
omega = generate_matsubara_frequencies(beta, n_max)
print(omega.shape)  # (201,)

# For plotting: iwn_vals is just for x-axis labels
iwn_vals = torch.arange(len(omega))  # [0, 1, 2, ..., 200]
# The actual frequencies are omega[iwn_vals] = iπ(2n+1)/β
```

---

#### `calculate_dos_range()`

Auto-calculate DOS range for interacting systems.

**File**: `manybody/preprocessing.py:88-125`

```python
calculate_dos_range(evals_min, evals_max, sigma_shift, U_max, margin=2.0)
```

**Parameters**:
- `evals_min` (float): Minimum eigenvalue from band structure
- `evals_max` (float): Maximum eigenvalue from band structure
- `sigma_shift` (float): Maximum |Re[Σ]| from self-energy
- `U_max` (float): Maximum Hubbard U value
- `margin` (float, optional): Additional safety margin (default: 2.0)

**Returns**:
- `tuple[float, float]`: (omega_min, omega_max) - Energy range for DOS calculation

**Formula**:
```
width = (evals_max - evals_min) + 2 * sigma_shift + U_max
center = (evals_min + evals_max) / 2
omega_min = center - width / 2 - margin
omega_max = center + width / 2 + margin
```

**Example**:
```python
from condmatTensor.manybody import calculate_dos_range

evals_min, evals_max = -3.0, 5.0  # Kagome-F bands
sigma_shift = 1.5  # From self-energy
U_max = 4.0  # Hubbard U on f-orbital

omega_min, omega_max = calculate_dos_range(
    evals_min, evals_max, sigma_shift, U_max
)
print(f"DOS range: [{omega_min:.1f}, {omega_max:.1f}]")
# Output: DOS range: [-12.5, 14.5]
```

---

#### `pauli_matrices()`

Return Pauli matrices as complex tensors.

**File**: `manybody/magnetic.py:795-812`

```python
pauli_matrices(device=None)
```

**Parameters**:
- `device` (torch.device, optional): Device for output tensors

**Returns**:
- `tuple`: (σx, σy, σz) - Three 2x2 complex matrices

**Example**:
```python
from condmatTensor.manybody.magnetic import pauli_matrices

sigma_x, sigma_y, sigma_z = pauli_matrices()
print(sigma_x)  # [[0, 1], [1, 0]]
```

---

### Classes

#### `BareGreensFunction`

Bare (non-interacting) Green's function G₀(iωₙ).

**File**: `manybody/preprocessing.py:55-171`

```python
BareGreensFunction()
```

**Methods**:

| Method | Description |
|--------|-------------|
| `compute(Hk, beta, mu=0.0, n_max=100, device=None)` | Compute G₀ from k-space Hamiltonian |

**Returns**:
- `BaseTensor`: with labels=['iwn', 'orb_i', 'orb_j']

**Formula**:
G₀(k, iωₙ) = (iωₙ + μ - H(k))⁻¹

**Example**:
```python
from condmatTensor.manybody.preprocessing import BareGreensFunction

G0 = BareGreensFunction()
G0_tensor = G0.compute(Hk, beta=10.0, mu=0.0, n_max=100)
print(G0_tensor.shape)  # (201, N_orb, N_orb)
```

---

#### `SelfEnergy`

Self-energy Σ(iωₙ) for many-body interactions.

**File**: `manybody/preprocessing.py:173-240`

```python
SelfEnergy()
```

**Methods**:

| Method | Description |
|--------|-------------|
| `initialize_zero(iwn, n_orb, orbital_names=None, device=None)` | Initialize Σ = 0 |

**Returns**:
- `BaseTensor`: with labels=['iwn', 'orb_i', 'orb_j']

**Example**:
```python
from condmatTensor.manybody.preprocessing import SelfEnergy, generate_matsubara_frequencies

omega = generate_matsubara_frequencies(beta=10.0, n_max=100)
Sigma = SelfEnergy()
Sigma_tensor = Sigma.initialize_zero(omega, n_orb=6)
```

---

#### `SpectralFunction`

Spectral function A(ω) from Green's function.

**File**: `manybody/preprocessing.py:242-627`

```python
SpectralFunction()
```

**Methods**:

| Method | Description |
|--------|-------------|
| `from_eigenvalues(eigenvalues, omega, eta=0.02, device=None)` | Compute from eigenvalues |
| `from_matsubara(G_iwn, omega, eta=0.02, method='simple', device=None, beta=None, n_min=0, n_max=None)` | Compute from Matsubara |
| `compute_dos(A=None)` | Compute total DOS |
| `plot(ax=None, orbital=-1, **kwargs)` | Plot spectral function |

**Analytic Continuation Methods**:
- `method='simple'`: Direct substitution iωₙ → ω + iη (fast, approximate)
- `method='pade'`: Padé approximant with Vidberg-Serene algorithm (accurate)
- `method='bethe'`: Bethe lattice analytical solution (semi-elliptical DOS)
- `method='maxent'`: Maximum entropy (NOT YET IMPLEMENTED)

**Pade Parameters** (when `method='pade'`):
- `beta` (float, required): Inverse temperature
- `n_min` (int): Minimum Matsubara index (default: 0)
- `n_max` (int, optional): Maximum Matsubara index (default: N//2)

**Bethe Parameters** (when `method='bethe'`):
- `z` (float, optional): Coordination number (auto-detected from lattice if None)
- `t` (float, optional): Hopping parameter (default: 1.0)
- `lattice` (BravaisLattice, optional): For auto-detecting coordination number

**Reference**: Vidberg H.J. and Serene J.W., J. Low Temp. Phys. 29, 179 (1977)

**Example**:
```python
from condmatTensor.manybody.preprocessing import SpectralFunction
import torch

# Non-interacting spectral function
omega = torch.linspace(-3, 3, 1000)
spectral = SpectralFunction()
A_nonint = spectral.from_eigenvalues(evals, omega, eta=0.02)
dos_nonint = spectral.compute_dos(A_nonint)

# Interacting spectral function with Pade continuation
A_int = spectral.from_matsubara(
    G_iwn, omega, eta=0.05, method='pade',
    beta=10.0, n_min=0, n_max=32
)
dos_int = spectral.compute_dos(A_int)

# Bethe lattice semi-elliptical DOS
A_bethe = spectral.from_matsubara(
    G_iwn, omega, eta=0.05, method='bethe',
    z=6.0, t=1.0
)
```

---

#### `AnalyticContinuationMethod`

Abstract base class for analytic continuation methods.

**File**: `manybody/analytic_continuation.py:1-50`

```python
class AnalyticContinuationMethod(ABC)
```

**Abstract Methods**:

| Method | Description |
|--------|-------------|
| `continue_to_real_axis(G_iwn, omega, eta, **kwargs)` | Transform G(iωₙ) → G(ω + iη), return A(ω) |

**Purpose**: Provides modular interface for different analytic continuation methods.

**Implementations**:
- `SimpleContinuation`: Direct substitution (fast, approximate)
- `PadeContinuation`: Padé approximant (accurate)
- `BetheLatticeContinuation`: Semi-elliptical DOS (testing)
- `MaxEntContinuation`: Maximum entropy (future, raises NotImplementedError)

---

#### `SimpleContinuation`

Direct substitution iωₙ → ω + iη (fast, approximate).

**File**: `manybody/analytic_continuation.py:51-130`

```python
SimpleContinuation()
```

**Methods**:
- `continue_to_real_axis(G_iwn, omega, eta=0.05, **kwargs)` → A(ω)

**Warning**: Only valid for very smooth functions. Fails for systems with sharp spectral features. Use with caution!

---

#### `PadeContinuation`

Padé approximant (Vidberg-Serene continued fraction).

**File**: `manybody/analytic_continuation.py:131-270`

```python
PadeContinuation()
```

**Methods**:
- `continue_to_real_axis(G_iwn, omega, eta=0.05, beta=10.0, n_min=0, n_max=None, **kwargs)` → A(ω)

**Algorithm**:
```
C_M(z) = a₀ / (1 + a₁(z-z₀) / (1 + a₂(z-z₁) / (...)))
```

**Reference**: Vidberg H.J. and Serene J.W., J. Low Temp. Phys. 29, 179 (1977)

---

#### `BetheLatticeContinuation`

Bethe lattice analytical solution for semi-elliptical DOS.

**File**: `manybody/analytic_continuation.py:271-370`

```python
BetheLatticeContinuation()
```

**Methods**:
- `continue_to_real_axis(G_iwn, omega, eta=0.05, z=None, t=1.0, lattice=None, **kwargs)` → A(ω)

**Formula**:
```
G₀(ω) = 2(z-1)/t² [ω + t²/(2(z-1)) - sqrt((ω + t²/(2(z-1)))² - 4)]
```

**Use Case**: Testing DMFT implementations where lattice problem can be solved analytically.

---

#### `MaxEntContinuation`

Maximum entropy analytic continuation (NOT YET IMPLEMENTED).

**File**: `manybody/analytic_continuation.py:371-400`

```python
MaxEntContinuation()
```

**Note**: Raises NotImplementedError. Consider using MaxEnt package or TRIQS for MaxEnt continuation.

---

#### `create_continuation_method()`

Factory function for creating analytic continuation methods.

**File**: `manybody/analytic_continuation.py:403-430`

```python
create_continuation_method(method)
```

**Parameters**:
- `method` (str): Method name ('simple', 'pade', 'bethe', 'maxent')

**Returns**:
- `AnalyticContinuationMethod`: Instance of the corresponding method

**Raises**:
- `ValueError`: If method name is unknown

**Example**:
```python
from condmatTensor.manybody.analytic_continuation import create_continuation_method

# Create method instances
pade = create_continuation_method('pade')
bethe = create_continuation_method('bethe')

# Use the methods
A_pade = pade.continue_to_real_axis(G_iwn, omega, eta=0.05, beta=10.0)
A_bethe = bethe.continue_to_real_axis(G_iwn, omega, eta=0.05, z=6.0)
```

---

#### `LocalMagneticModel`

H = H₀ + J·S model using spinor approach.

**File**: `manybody/magnetic.py:39-674`

```python
LocalMagneticModel(H0=None, J=1.0, S_init=None)
```

**Parameters**:
- `H0` (BaseTensor or torch.Tensor): Initial Hamiltonian
- `J` (float): Exchange coupling strength
- `S_init` (torch.Tensor): Initial spin configuration

**Methods**:

| Method | Description |
|--------|-------------|
| `build_spinful_hamiltonian(H0_spinless, soc_tensor=None, lattice=None, device=None)` | Build spinful H |
| `add_magnetic_exchange(Hk, S_config, J=None, lattice=None)` | Add J·S term |
| `add_effective_magnetic_field(Hk, B_field, g_factor=2.0, mu_B=1.0, lattice=None)` | Add Zeeman term |
| `compute_green_function(Hk, omega_n, mu=0.0)` | Compute G(k, iωₙ) |
| `self_consistency_loop(beta, mixing=0.5, tol=1e-6, max_iter=100, n_max=100, mu=0.0, lattice=None, verbose=True)` | DMFT-like loop |

**Example**:
```python
from condmatTensor.manybody.magnetic import LocalMagneticModel

model = LocalMagneticModel(J=1.0)
Hk_spinful = model.build_spinful_hamiltonian(Hk_spinless, lattice=lattice)
```

---

#### `KondoLatticeSolver`

Specialized solver for Kondo lattice models.

**File**: `manybody/magnetic.py:676-764`

```python
KondoLatticeSolver(H0=None, J=1.0, S_init=None)
```

**Methods**:

| Method | Description |
|--------|-------------|
| `estimate_kondo_temperature(density_of_states, J=None)` | Estimate T_K |
| `compute_rkky_interaction(Hk, q_vector)` | Compute RKKY coupling |

**Note**: RKKY computation returns 0.0 placeholder (not implemented).

---

#### `SpinFermionModel`

General spin-fermion coupling model.

**File**: `manybody/magnetic.py:766-793`

```python
SpinFermionModel(H0=None, J_tensor=None, S_init=None)
```

**Parameters**:
- `H0` (BaseTensor or torch.Tensor): Initial Hamiltonian
- `J_tensor` (torch.Tensor): Tensor of exchange couplings
- `S_init` (torch.Tensor): Initial spin configuration

---

#### `ImpuritySolverABC`

Abstract base class for DMFT impurity solvers.

**File**: `manybody/impSolvers/base.py:1-90`

```python
class ImpuritySolverABC(ABC)
```

**Abstract Methods**:

| Method | Description |
|--------|-------------|
| `solve(G_input, **kwargs)` | Solve impurity problem, return Σ |
| `solver_name` (property) | Return solver name string |
| `supported_orbitals` (property) | Return max orbitals (-1=unlimited) |

**Purpose**: Enables polymorphic use of different impurity solvers (IPT, ED, NRG, CTQMC) in DMFT loop.

**Example**:
```python
from condmatTensor.manybody import IPTSolver, ImpuritySolverABC

solver = IPTSolver(beta=10.0, n_max=100)
assert isinstance(solver, ImpuritySolverABC)  # Type checking
Sigma = solver.solve(G_input)
```

---

#### `IPTSolver`

Iterated Perturbation Theory solver for DMFT (second-order).

**File**: `manybody/impSolvers/ipt.py:1-420`

```python
IPTSolver(beta, n_max=100, device=None)
```

**Parameters**:
- `beta` (float): Inverse temperature
- `n_max` (int): Maximum Matsubara frequency index
- `device` (torch.device, optional): Device for computation

**Methods**:

| Method | Description |
|--------|-------------|
| `solve(G_input, max_iter=1, tol=1e-6, **kwargs)` | Compute Σ using TRIQS formula |
| `_validate_self_energy(Sigma, U_orb)` | Validate Σ against physical constraints (internal) |
| `solver_name` (property) | Returns "IPT" |
| `supported_orbitals` (property) | Returns -1 (unlimited) |
| `G_loc` (property) | Local G from last solve() |
| `Sigma` (property) | Σ from last solve() |

**Formula** (TRIQS-style imaginary time):
```
Σ(τ) = U² · G₀(τ)³
Σ(iωₙ) = F[Σ(τ)]
```

**Matsubara Frequency Transforms**:
The solver uses proper fermionic Matsubara frequency transforms (not generic PyTorch FFT):

```
G(iωₙ) = (1/β) ∫₀^β dτ e^(iωₙτ) G(τ)  ← Fourier transform
G(τ) = (1/β) Σₙ e^(-iωₙτ) G(iωₙ)      ← Inverse Fourier transform
where ωₙ = π(2n+1)/β (fermionic Matsubara frequencies)
```

**Numerical Stabilization**:
- Adaptive clipping: `g_clip = max(3.0, 10.0 / U_orb[i])`
- For U=4, clip=3 gives max Σ ~ 16 × 27 = 432
- For U=2, clip=5 gives max Σ ~ 4 × 125 = 500
- Complex G(τ) preserved: clips magnitude but keeps phase

**Self-Energy Validation**:
The `_validate_self_energy()` method checks:
1. High-frequency limit: Σ(iωₙ) → U²n/β as ωₙ → ∞
2. Magnitude should be O(1) to O(100) for typical U values
3. Orbital selectivity: Σ_f >> Σ_d for U_f >> U_d

**Orbital-dependent U**: Read from `OrbitalMetadata.U`
- U_d ≈ 0-1 (conductive d-orbitals)
- U_f ≈ 4-8 (localized f-orbitals)
- For U_f = 4.0, expected |Im Σ_f(iω₀)| ~ O(10) to O(200)
- For U_f = 2.0, expected |Im Σ_f(iω₀)| ~ O(1) to O(50)

**Example**:
```python
from condmatTensor.manybody import IPTSolver
from condmatTensor.core.types import OrbitalMetadata

# Create solver with orbital-dependent U
solver = IPTSolver(beta=10.0, n_max=64)
G_input.orbital_metadatas = [
    OrbitalMetadata(site='K1', orb='d', local=False, U=0.5),
    OrbitalMetadata(site='F', orb='f', local=True, U=4.0),
]
Sigma = solver.solve(G_input)
```

---

#### `SingleSiteDMFTLoop`

DMFT self-consistency loop with polymorphic impurity solver support.

**File**: `manybody/dmft.py:1-400`

```python
SingleSiteDMFTLoop(Hk, omega, solver, mu=0.0, mixing=0.5, verbose=True)
```

**Parameters**:
- `Hk` (BaseTensor): k-space Hamiltonian, labels=['k', 'orb_i', 'orb_j']
- `omega` (torch.Tensor): Matsubara frequency grid
- `solver` (ImpuritySolverABC): Impurity solver (IPT, ED, NRG, etc.)
- `mu` (float): Chemical potential (default: 0.0)
- `mixing` (float or MixingMethod): Mixing parameter (0 < mixing <= 1)
- `verbose` (bool): Print iteration progress

**Methods**:

| Method | Description |
|--------|-------------|
| `run(max_iter=100, tol=1e-6, Sigma_init=None)` | Run DMFT loop |
| `Sigma` (property) | Final converged self-energy |
| `G_loc` (property) | Final local Green's function |
| `G0` (property) | Final Weiss field |
| `n_iterations` (property) | Iterations to converge |
| `get_convergence_history()` | Dict with 'Sigma_diff', 'Sigma_norm' |

**DMFT Algorithm (7 steps)**:
1. Start with Σ = 0 (non-interacting)
2. Compute G(k,iω) = [iω+μ-H(k)-Σ]⁻¹
3. Extract G_loc = (1/N_k) Σ_k G(k,iω)
4. Compute Weiss field: G₀⁻¹ = G_loc⁻¹ + Σ
5. Solve impurity: Σ_new = solver.solve(G₀)
6. Mix: Σ = (1-α)·Σ_old + α·Σ_new
7. Check convergence: |ΔΣ|/|Σ| < tol

**Example**:
```python
from condmatTensor.manybody import IPTSolver, SingleSiteDMFTLoop
from condmatTensor.manybody.preprocessing import generate_matsubara_frequencies

# Setup
omega = generate_matsubara_frequencies(beta=10.0, n_max=64)
solver = IPTSolver(beta=10.0, n_max=64)
dmft = SingleSiteDMFTLoop(Hk_mesh, omega, solver, mixing=0.5)

# Run
Sigma = dmft.run(max_iter=50, tol=1e-5)
print(f"Converged in {dmft.n_iterations} iterations")
```

---

#### `MixingMethod` / `LinearMixing`

Mixing strategies for self-energy updates.

**File**: `manybody/dmft.py:25-80`

```python
LinearMixing(alpha=0.5)
```

**Parameters**:
- `alpha` (float): Mixing parameter (0 < alpha <= 1)

**Formula**:
```
Σ_mixed = (1-α)·Σ_old + α·Σ_new
```

**Future**: AndersonMixing, BayesianMixing (via LEVEL 7 optimization)

---

## LEVEL 5: Analysis

**Status**: ⚠️ Partial (Topology, QGT not implemented)
**Path**: `src/condmatTensor/analysis/`
**Files**: `dos.py` (601 lines), `bandstr.py` (626 lines), `plotting_style.py` (87 lines)

### Modules

#### `plotting_style`

Standardized plotting style constants for publication-quality plots.

**File**: `analysis/plotting_style.py:1-70`

**Constants**:

| Constant | Description |
|----------|-------------|
| `DEFAULT_FIGURE_SIZES` | Dict of predefined figure sizes ('single', 'dual', 'triple', '2x2', 'band_dos', etc.) |
| `DEFAULT_COLORS` | Dict of color scheme (primary, reference, secondary, etc.) |
| `DEFAULT_FONTSIZES` | Dict of font sizes for labels, titles, legend, etc. |
| `DEFAULT_STYLING` | Dict of styling options (grid_alpha, band_linewidth, dpi, etc.) |
| `DEFAULT_COLORMAPS` | Dict of colormaps (orbital_weight, spin, orbitals, etc.) |
| `LINE_STYLES` | Dict of line styles (solid, dashed, dotted, dash_dot) |
| `MARKER_STYLES` | Dict of marker styles (circle, square, triangle, etc.) |

**Example**:
```python
from condmatTensor.analysis import DEFAULT_COLORS, DEFAULT_FIGURE_SIZES
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZES['single'])
ax.plot(x, y, color=DEFAULT_COLORS['primary'])
```

---

### Classes

#### `DOSCalculator`

Density of States calculator with Lorentzian broadening.

**File**: `analysis/dos.py:28-602`

```python
DOSCalculator()
```

**Methods**:

| Method | Description |
|--------|-------------|
| `from_eigenvalues(E_k, omega, eta=0.02)` | Compute DOS from eigenvalues |
| `from_spectral_function(A, omega)` | Compute DOS from spectral function |
| `from_matsubara_pade(G_iwn, omega, eta=0.05, beta=None, n_min=0, n_max=None)` | Compute DOS from Matsubara G using Pade |
| `plot(ax=None, energy_range=None, ylabel, xlabel, title, fontsize=12, fill=True, **kwargs)` | Plot DOS |
| `plot_with_reference(reference_energies, labels=None, colors=None, linestyles=None, ax=None, energy_range=None, ylabel, xlabel, title, fontsize=12, legend=True, **kwargs)` | Plot DOS with vertical reference lines |
| `plot_comparison(other_dos_data, labels, colors=None, alpha=0.7, ax=None, energy_range=None, ylabel, xlabel, title, fontsize=12, linewidth=1.5, legend=True, fill=False, **kwargs)` | Overlay multiple DOS curves |
| `plot_multi_panel(dos_list, titles, figsize=None, energy_range=None, ylabel, xlabel, fontsize=12, sharex=True, sharey=True, **kwargs)` | Create multi-panel DOS comparison |

**Formula**:
DOS(ω) = Σ_k (1/π) * η / [(ω - E_k)² + η²]

**Pade Parameters** (for `from_matsubara_pade`):
- `G_iwn`: Green's function on Matsubara frequencies (labels=['iwn', 'orb_i', 'orb_j'])
- `omega`: Real frequency grid for output
- `eta`: Imaginary shift for analytic continuation (default: 0.05)
- `beta` (float, required): Inverse temperature
- `n_min` (int): Minimum Matsubara index (default: 0)
- `n_max` (int, optional): Maximum Matsubara index (default: N//2)

**Examples**:
```python
from condmatTensor.analysis import DOSCalculator
import torch

omega = torch.linspace(-3, 3, 1000)
dos_calc = DOSCalculator()
dos = dos_calc.from_eigenvalues(evals, omega, eta=0.02)
dos_calc.plot()

# Plot with reference line for flat band
dos_calc.plot_with_reference(-2.0, label='Flat band')

# Compare two DOS curves
dos_calc.plot_comparison((omega2, rho2), labels=['Model 1', 'Model 2'])

# Compute DOS from interacting Green's function using Pade
dos_int = dos_calc.from_matsubara_pade(
    G_iwn, omega, eta=0.05, beta=10.0, n_min=0, n_max=32
)
```

---

#### `ProjectedDOS`

Projected DOS onto specific orbitals.

**File**: `analysis/dos.py:364-522`

```python
ProjectedDOS()
```

**Methods**:

| Method | Description |
|--------|-------------|
| `from_eigenvalues(E_k, U, omega, eta=0.02, orbital_labels=None)` | Compute PDOS |
| `get_projected_dos()` | Get PDOS values |
| `plot_projected(ax=None, energy_range=None, ylabel, xlabel, title, fontsize=12, stacked=True, **kwargs)` | Plot PDOS |

**Example**:
```python
from condmatTensor.analysis import ProjectedDOS

pdos_calc = ProjectedDOS()
pdos = pdos_calc.from_eigenvalues(evals, evecs, omega, eta=0.02, orbital_labels=['A', 'B', 'C'])
pdos_calc.plot_projected(stacked=True)
```

---

#### `BandStructure`

Band structure calculator with publication-quality plotting methods.

**File**: `analysis/bandstr.py:20-550`

```python
BandStructure()
```

**Methods**:

| Method | Description |
|--------|-------------|
| `compute(eigenvalues, k_path, ticks=None)` | Store band structure results |
| `plot(ax=None, energy_range=None, ylabel, title, fontsize=12, **kwargs)` | Plot band structure |
| `plot_with_dos(eigenvalues_mesh, omega, eta=0.02, energy_range=None, ylabel, dos_xlabel, title, fontsize=12, figsize=(10, 5), dos_color='skyblue', **kwargs)` | Plot bands + DOS |
| `plot_colored_by_weight(eigenvectors, orbital_indices, ax=None, cmap='viridis', s=10, vmin=0, vmax=1, ylabel, xlabel, title, fontsize=12, colorbar=True, colorbar_label='Orbital weight', **kwargs)` | Plot bands colored by orbital weight |
| `add_reference_line(energy, label=None, color='red', linestyle='--', alpha=0.7, ax=None, **kwargs)` | Add horizontal reference line |
| `plot_comparison(other_eigenvalues, labels, colors=None, alpha=0.6, ax=None, energy_range=None, ylabel, title, fontsize=12, linewidth=1.0, legend=True, **kwargs)` | Overlay multiple band structures |
| `plot_multi_panel(eigenvalues_list, titles, k_paths=None, ticks_list=None, figsize=None, energy_range=None, ylabel, fontsize=12, **kwargs)` | Create multi-panel comparison |

**Examples**:
```python
from condmatTensor.analysis import BandStructure

# Basic band structure
bs = BandStructure()
bs.compute(evals, k_path, ticks=[0, 50, 100, 150])
bs.plot()

# Add reference line for flat band
ax = bs.plot()
bs.add_reference_line(-2.0, label='Flat band')
plt.legend()

# Color by f-orbital weight
bs.plot_colored_by_weight(eigenvectors, orbital_indices=[6, 7])

# Compare full vs effective model
bs.plot_comparison(evals_eff, labels=['Full', 'Effective'])

# Multi-panel comparison
bs.plot_multi_panel([evals1, evals2, evals3], titles=['t_f=-1.0', 't_f=-0.5', 't_f=0.0'])
```
```

---

## LEVEL 7: Optimization

**Status**: ⚠️ Partial (ML interface not implemented)
**Path**: `src/condmatTensor/optimization/`
**Files**: `bayesian/__init__.py` (466 lines), `magnetic.py` (631 lines)

### Backend Priority

The `BayesianOptimizer` supports multiple backends with automatic fallback:

| Priority | Backend | Package | Description |
|----------|---------|---------|-------------|
| **1 (Preferred)** | SOBER | `sober-bo==2.0.4` | Sequential Optimization using Ensemble of Regressors |
| **2** | BoTorch | `botorch>=0.9.0` | Gaussian Process with Expected Improvement |
| **3 (Fallback)** | Simple | Python stdlib | Thompson sampling / random search |

---

### Classes

#### `BayesianOptimizer`

Bayesian optimization for hyperparameter tuning.

**File**: `optimization/bayesian/__init__.py:32-305`

```python
BayesianOptimizer(bounds, n_init=10, n_iter=50, backend='auto', seed=None)
```

**Parameters**:
- `bounds` (list[tuple[float, float]]): List of (min, max) for each parameter
- `n_init` (int): Number of initial random points (default: 10)
- `n_iter` (int): Number of optimization iterations (default: 50)
- `backend` (str): 'sober', 'botorch', 'simple', or 'auto' (default: 'auto')
- `seed` (int, optional): Random seed for reproducibility

**Methods**:

| Method | Description |
|--------|-------------|
| `optimize(objective, maximize=False, verbose=True, device=None)` | Run optimization |
| `get_best()` | Get best observation (X, y) |
| `reset()` | Reset optimizer state |

**Returns**:
- `tuple`: (X_best, y_best) - Best parameters and objective value

**Example**:
```python
from condmatTensor.optimization import BayesianOptimizer

def objective_fn(x):
    """Objective function to minimize."""
    return (x[0] - 0.5)**2 + (x[1] - 0.3)**2

opt = BayesianOptimizer(
    bounds=[(0, 1), (0, 1)],
    backend='auto'
)
X_best, y_best = opt.optimize(objective_fn, n_init=10, n_iter=50, device=device)
print(f"Best: {X_best}, {y_best}")
```

---

#### `MultiObjectiveOptimizer`

Multi-objective Bayesian optimization with Pareto front.

**File**: `optimization/bayesian/__init__.py:307-454`

```python
MultiObjectiveOptimizer(bounds, n_objectives, n_init=20, n_iter=100, seed=None)
```

**Parameters**:
- `bounds` (list[tuple[float, float]]): Parameter bounds
- `n_objectives` (int): Number of objectives
- `n_init` (int): Initial random points (default: 20)
- `n_iter` (int): Optimization iterations (default: 100)
- `seed` (int, optional): Random seed

**Methods**:

| Method | Description |
|--------|-------------|
| `optimize(objective, verbose=True, device=None)` | Run multi-objective optimization |
| `_get_pareto_front()` | Extract Pareto-optimal solutions |

---

#### `EffectiveArrayOptimizer`

Downfolding high-dimensional magnetic systems to effective models.

**File**: `optimization/magnetic.py:31-620`

```python
EffectiveArrayOptimizer(H_cc_0, H_full, method='eigenvalue', f_orbital_indices=None, lattice=None)
```

**Parameters**:
- `H_cc_0` (BaseTensor): Conduction band Hamiltonian
- `H_full` (BaseTensor): Full Hamiltonian including f-orbitals
- `method` (str): 'eigenvalue' or 'perturbation' (default: 'eigenvalue')
- `f_orbital_indices` (list[int], optional): Indices of f-orbitals
- `lattice` (BravaisLattice, optional): Lattice for k-space operations

**Methods**:

| Method | Description |
|--------|-------------|
| `optimize(J_bounds=(0.01, 10.0), S_bounds=None, n_init=20, n_iter=100, backend='auto', verbose=True, device=None)` | Find effective J and S |
| `verify(k_path=None, verbose=True)` | Compare band structures |
| `perturbation_theory(epsilon_f, V_cf)` | Schrieffer-Wolff transformation |
| `plot_comparison(k_path=None, ticks=None, ax=None)` | Plot comparison |

**Example**:
```python
from condmatTensor.optimization import EffectiveArrayOptimizer

opt = EffectiveArrayOptimizer(H_cc_0, H_full, lattice=lattice)
J_eff, S_eff = opt.optimize(J_bounds=(0.1, 5.0), n_iter=100)
opt.verify()
```

---

#### `SoberBackend`

SOBER (Sequential Optimization using Ensemble of Regressors) backend.

**File**: `optimization/bayesian/sober_backend.py:22-212`

**Reference**: https://github.com/ma921/SOBER

---

#### `BotorchBackend`

BoTorch Gaussian Process backend.

**File**: `optimization/bayesian/botorch_backend.py:21-179`

**Reference**: https://botorch.org/

---

#### `SimpleBackend`

Fallback backend using Thompson sampling or random search.

**File**: `optimization/bayesian/simple_backend.py:17-137`

---

### Functions

#### `latin_hypercube_sampling()`

Latin Hypercube Sampling for initial Bayesian optimization points.

**File**: `optimization/bayesian/utils.py:13-59`

```python
latin_hypercube_sampling(bounds, n_samples, device, seed=None)
```

**Parameters**:
- `bounds` (list[tuple[float, float]]): Parameter bounds
- `n_samples` (int): Number of samples
- `device` (torch.device): Device for output
- `seed` (int, optional): Random seed

**Returns**:
- `torch.Tensor`: Shape (n_samples, dim)

---

#### `run_sober_optimization()`

Convenience function for SOBER optimization.

**File**: `optimization/bayesian/sober_backend.py:214-352`

---

#### `run_botorch_optimization()`

Convenience function for BoTorch optimization.

**File**: `optimization/bayesian/botorch_backend.py:181-277`

---

#### `run_simple_optimization()`

Convenience function for simple backend optimization.

**File**: `optimization/bayesian/simple_backend.py:139-236`

---

## Import Reference

```python
# LEVEL 1: Core
from condmatTensor.core import BaseTensor, OrbitalMetadata, get_device

# LEVEL 2: Lattice
from condmatTensor.lattice import (
    BravaisLattice,
    HoppingModel,
    generate_kmesh,
    generate_k_path
)

# LEVEL 3: Solvers
from condmatTensor.solvers import diagonalize

# LEVEL 4: Many-Body
from condmatTensor.manybody.preprocessing import (
    generate_matsubara_frequencies,
    BareGreensFunction,
    SelfEnergy,
    SpectralFunction
)
from condmatTensor.manybody.magnetic import (
    LocalMagneticModel,
    KondoLatticeSolver,
    SpinFermionModel,
    pauli_matrices
)
# Impurity solvers (ABC + implementations)
from condmatTensor.manybody.impSolvers import (
    ImpuritySolverABC,
    IPTSolver,
)
# DMFT loop
from condmatTensor.manybody.dmft import (
    SingleSiteDMFTLoop,
    MixingMethod,
    LinearMixing,
)

# LEVEL 5: Analysis
from condmatTensor.analysis import (
    DOSCalculator,
    ProjectedDOS,
    BandStructure,
    # Plotting style constants
    DEFAULT_FIGURE_SIZES,
    DEFAULT_COLORS,
    DEFAULT_FONTSIZES,
    DEFAULT_STYLING,
    DEFAULT_COLORMAPS,
    LINE_STYLES,
    MARKER_STYLES,
)

# LEVEL 7: Optimization
from condmatTensor.optimization import (
    BayesianOptimizer,
    MultiObjectiveOptimizer,
    EffectiveArrayOptimizer
)
```

---

## Complete Workflow Example

```python
import torch
from condmatTensor.core import get_device
from condmatTensor.lattice import BravaisLattice, HoppingModel, generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure, DOSCalculator

# Get device
device = get_device()

# Create Kagome lattice
cell_vectors = torch.tensor([
    [0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0],
    [0.0, 0.0, 1.0]
], device=device)
basis_positions = [(0, 0), (0.5, 0), (0.25, 0.25)]
lattice = BravaisLattice(cell_vectors, basis_positions, [1, 1, 1])

# Build tight-binding model
model = HoppingModel(lattice, orbital_labels=['A', 'B', 'C'])
model.add_hopping('A', 'B', [0, 0], 1.0)
model.add_hopping('B', 'C', [0, 0], 1.0)
model.add_hopping('C', 'A', [0, 0], 1.0)

# Generate k-path and diagonalize
k_path, ticks = generate_k_path(lattice, ['G', 'K', 'M', 'G'], n_per_segment=50)
Hk = model.build_Hk(k_path)
evals, evecs = diagonalize(Hk.tensor)

# Plot band structure
bs = BandStructure()
bs.compute(evals.cpu().numpy(), k_path.cpu().numpy(), ticks)
bs.plot(tick_labels=['Gamma', 'K', 'M', 'Gamma'])
```

---

## Validation Examples

The `examples/` directory demonstrates and validates core physics:

1. **`kagome_bandstructure.py`** - Pure Kagome lattice
   - Expected: flat band at E=-2|t|, Dirac points at K

2. **`kagome_with_f_bandstructure.py`** - Kagome-F (4 sites/cell)

3. **`kagome_spinful_bandstructure.py`** - Spinful systems with Zeeman coupling

4. **`kagome_f_effective_array.py`** - Bayesian downfolding

5. **`gpu_performance_benchmark.py`** - GPU performance benchmark

---

*Last Updated: 2026-02-03*
