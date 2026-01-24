# condmatTensor Architecture Plan

> **For detailed dependency analysis and comparison with reference libraries (NumPy, WannierTools, TRIQS), see `DEPENDENCY_ANALYSIS.md` in this directory.**

## Overview

A PyTorch-based condensed matter physics library with a unified `BaseTensor` class that handles all tensor operations (Hamiltonian, Green's functions, etc.). The library focuses on tensor-first design, GPU acceleration, and modular workflows.

**Kagome Lattice Examples**: Run BOTH test examples after module completion to validate:

1. **Pure Kagome** (`examples/kagome_bandstructure.py`): Triangular lattice with corner-sharing triangles (3 sites per unit cell). Tests basic tight-binding, diagonalization, band structure. Expected: flat band at E = -2|t|, Dirac points at K. **463 lines**.

2. **Kagome-F** (`examples/kagome_with_f_bandstructure.py`): Kagome lattice with central f-orbital atom (4 sites per unit cell). Models heavy-fermion systems. Tests multi-orbital physics, orbital hybridization, parameter scans (t, t_f, ε_f). **556 lines**.

**Implementation Status** (as of 2026-01-23):
- **LEVEL 1-3, partial LEVEL 4, LEVEL 5, and LEVEL 7**: ~3,580 lines implemented (excluding __init__.py)
- Examples validate 3 methods of Hamiltonian construction agree to <1e-6:
  1. Direct k-space (analytic formula)
  2. `TightBindingModel.build_Hk()` (direct from hopping)
  3. `TightBindingModel.build_HR().to_k_space()` (real-space + Fourier)

## Simplified Module Structure

```
src/condmatTensor/
├── __init__.py                 # Package initialization (v0.0.1), clear dependency relations
│
├── core/                       [LEVEL 1: Foundation - 0 internal deps] ✅ IMPLEMENTED
│   ├── __init__.py             # Exports: BaseTensor
│   ├── base.py (138 lines)     # BaseTensor class (tensor + labels + orbital names + displacements)
│   ├── math.py                 # tensor math utilities, Berry curvature helpers [NOT IMPLEMENTED]
│   └── gpu_utils.py            # device selection, memory estimates, chunking [NOT IMPLEMENTED]
│
├── lattice/                    [LEVEL 2: Data Structures - +LEVEL 1] ✅ IMPLEMENTED
│   ├── __init__.py             # Exports: BravaisLattice, TightBindingModel, generate_kmesh, generate_k_path
│   ├── model.py (337 lines)    # BravaisLattice class, TightBindingModel class
│   ├── bzone.py (94 lines)     # generate_kmesh(), generate_k_path(), k_frac_to_cart()
│   └── symmetry.py             [LEVEL 10: Symmetry Reduction - +LEVEL 1, LEVEL 2] [NOT IMPLEMENTED]
│
├── solvers/                    [LEVEL 3: Computational Engines - +LEVEL 1, LEVEL 2] ✅ PARTIAL
│   ├── __init__.py             # Exports: diagonalize
│   ├── diag.py (36 lines)      # diagonalize() function
│   └── ed_cnn.py               # CNN-self attention CI selection [NOT IMPLEMENTED, will use manybody.ed]
│
├── manybody/                   [LEVEL 4: Many-Body Physics - +LEVEL 1, LEVEL 2, LEVEL 3] ✅ PARTIAL
│   ├── __init__.py             # Exports: preprocessing and magnetic classes
│   ├── preprocessing.py (496 lines)  ✅ Matsubara frequencies, BareGreensFunction, SelfEnergy, SpectralFunction
│   ├── magnetic.py (~870 lines)    ✅ LocalMagneticModel, KondoLatticeSolver, SpinFermionModel, pauli_matrices
│   │   ├── **FIXED (2024-01-24)**: `build_spinful_hamiltonian()` now correctly copies both intra-site AND inter-site hopping
│   │   ├── **References cited**: Kondo/s-d model literature for on-site J@S coupling
│   │   └── **New methods**: `_is_already_spinful()`, updated `_detect_f_orbitals()` for spinful systems
│   ├── dmft.py                 # SingleSiteDMFTLoop class [NOT IMPLEMENTED]
│   ├── cdmft.py                # ClusterDMFTLoop class (CDMFT/DCA style) [NOT IMPLEMENTED]
│   ├── ipt.py                  # IPT solver [NOT IMPLEMENTED]
│   └── ed.py                   # ED solver [NOT IMPLEMENTED]
│
├── analysis/                   [LEVEL 5: Observables - +LEVEL 1, LEVEL 2, LEVEL 3] ✅ PARTIAL
│   ├── __init__.py             # Exports: BandStructure, DOSCalculator, ProjectedDOS
│   ├── dos.py (312 lines)      # DOSCalculator class, ProjectedDOS class
│   ├── qgt.py                  # QGT functions: hk_to_g_layer, hr_to_g_layer, compute_qgt [NOT IMPLEMENTED]
│   ├── topology.py             # Berry curvature, Chern, Z₂, AHE functions [NOT IMPLEMENTED]
│   ├── fermi.py                # FermiSurface class [NOT IMPLEMENTED]
│   └── bandstr.py (237 lines)  # BandStructure class with plot() and plot_with_dos()
│
├── transport/                  [LEVEL 6: Transport Properties - +LEVEL 1, LEVEL 2] ❌ NOT IMPLEMENTED
│   ├── __init__.py
│   ├── rgf.py                  # RGF class and stacking_rgf() function
│   └── transport.py            # Transport class with transmission(), conductivity()
│
├── optimization/               [LEVEL 7: Parameter Search - +LEVEL 4] ✅ PARTIAL
│   ├── __init__.py             # Exports: BayesianOptimizer, MultiObjectiveOptimizer, EffectiveArrayOptimizer
│   ├── bayesian.py (573 lines)      ✅ BayesianOptimizer, MultiObjectiveOptimizer (uses botorch/scikit-optimize)
│   ├── magnetic.py (~620 lines)     ✅ EffectiveArrayOptimizer for Kondo/spin-fermion model downfolding
│   │   ├── **FIXED (2024-01-24)**: Proper spinful system handling with `_is_already_spinful()` detection
│   │   ├── **FIXED (2024-01-24)**: Updated `_detect_f_orbitals()` to detect both f_up and f_down orbitals
│   │   ├── **FIXED (2024-01-24)**: J@S coupling now properly on-site (per lattice site)
│   │   └── **References cited**: Kondo breakdown, orbital-selective Kondo models
│   └── ml_interface.py         # ML surrogate models [NOT IMPLEMENTED]
│
├── interface/                  [LEVEL 8: External I/O - +LEVEL 1, LEVEL 2] ❌ NOT IMPLEMENTED
│   ├── __init__.py
│   ├── yaml_reader.py          # YAMLReader class for lattice/H input
│   └── wannier_reader.py       # WannierReader class for hr.dat → BaseTensor
│
└── logging/                    [LEVEL 9: Utilities - 0 internal deps] ❌ NOT IMPLEMENTED
    └── __init__.py              # CalculationLogger

**Logging Features:** [NOT IMPLEMENTED]
- Input parameters tracking
- DMFT iteration metrics (convergence, self-energy changes)
- Tensor shapes, devices (CPU/CUDA), memory usage
- Timing information per calculation block
- Results output to JSON/HDF5 for reproducibility
- Bug/mistake detection and reporting

└── lattice/symmetry/           [LEVEL 10: Symmetry Reduction - +LEVEL 1, LEVEL 2] [NOT IMPLEMENTED]
    └── __init__.py             # SymmetryReducer2D, SymmetryReducer3D, SymmetryReducer (after interface complete)

**Symmetry Features:**
- **2D**: Self-implemented point group detection (C_n, D_n) from cell_vectors
- **3D**: spglib space group detection (all 230 space groups)
- Irreducible Brillouin zone (IBZ) generation
- Symmetry-aware k-mesh with weights
- Time-reversal symmetry exploitation
- Memory savings via symmetric summation
```

### Dependency Graph (Visual)

```
                    EXTERNAL DEPENDENCIES
                 torch | numpy | matplotlib | scipy
                      └───────┴──────────────┴────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                     │
                ▼                                     ▼
    ┌─────────────────────┐               ┌─────────────────────┐
    │   LEVEL 1: Core     │               │   LEVEL 9: Logging  │
    │  • base.py          │               │   (independent)     │
    │  • math.py          │               │                     │
    │  • gpu_utils.py     │               │                     │
    └─────────┬───────────┘               └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  LEVEL 2: Lattice   │
    │  • model.py         │
    │  • bzone.py         │
    └─────────┬───────────┘
              │
    ┌─────────┴─────────────────────────────────────┐
    │                                               │
    ▼                                               ▼
┌──────────────────┐                        ┌──────────────────┐
│ LEVEL 3: Solvers │                        │ LEVEL 8: Interface│
│  • diag.py       │                        │  • yaml_reader   │
│  • ipt.py        │                        │  • wannier_reader│
│  • ed.py         │                        └─────────┬────────┘
│  • ed_cnn.py     │                                  │
└────────┬─────────┘                                  │
         │                                            ▼
         ▼                                  ┌───────────────────┐
┌───────────────────────┐         ┌───────────────────┐  │ LEVEL 10: Symmetry│
│ LEVEL 4: Many-Body    │         │ LEVEL 5: Analysis │  │  • symmetry.py    │
│  • preprocessing.py   │         │  • dos.py         │  │  (IBZ reduction)  │
│  • magnetic.py        │         │  • bandstr.py     │  └─────────┬─────────┘
│  • dmft.py (DMFT)     │         │  • fermi.py       │            │
│  • cdmft.py (CDMFT)   │         │  • qgt.py         │            │
│  • ipt.py, ed.py      │         │  • topology.py    │            │
└───────────┬───────────┘         └─────────┬─────────┘            │
            │                       │  • topology.py   │            │
            ▼                       └─────────┬─────────┘            │
┌───────────────────────┐                     │                    │
│ LEVEL 7: Optimization  │                     │                    │
│  • bayesian.py        ◄─────────────────────┘                    │
│  • ml_interface.py    │                                          │
└───────────────────────┘                                          │
         ┌───────────────────┐                                      │
         │ LEVEL 6: Transport │                                      │
         │  • rgf.py         │◄──────┐                              │
         │  • transport.py   │       │                              │
         └───────────────────┘       │                              │
                                      │                              │
         ┌───────────────────┐       │                              │
         │ LEVEL 2: Lattice  │───────┘──────────────────────────────┘
         └───────────────────┘
```

---

## Module Dependencies (Detailed)

This section provides detailed dependency information for each level, including what can be imported independently and the specific internal/external dependencies.

### LEVEL 1: Core (No Internal Dependencies)

**Files**: `core/base.py`, `core/math.py` (not implemented), `core/gpu_utils.py` (not implemented)

**Internal Dependencies**: None

**External Dependencies**: `torch`

**Can Import Independently**: ✅ Yes

**Key Classes**:
- `BaseTensor` - Unified tensor representation with semantic labels
- `get_device()` - Auto-detect CUDA/CPU/MPS device

**Usage**:
```python
from condmatTensor.core import BaseTensor, get_device
```

---

### LEVEL 2: Lattice (Depends on LEVEL 1)

**Files**: `lattice/model.py`, `lattice/bzone.py`

**Internal Dependencies**: `core` (BaseTensor, get_device)

**External Dependencies**: `torch`, `numpy`

**Can Import Independently**: ✅ Yes (after core)

**Key Classes**:
- `BravaisLattice` - Bravais lattice with multiple sites per unit cell
- `TightBindingModel` - General tight-binding model builder
- `generate_kmesh()` - Generate uniform k-mesh in fractional coordinates
- `generate_k_path()` - Generate k-path along high-symmetry lines

**Usage**:
```python
from condmatTensor.lattice import (
    BravaisLattice,
    TightBindingModel,
    generate_kmesh,
    generate_k_path
)
```

---

### LEVEL 3: Solvers (Depends on LEVEL 1, LEVEL 2)

**Files**: `solvers/diag.py`, `solvers/ipt.py` (not implemented), `solvers/ed.py` (not implemented), `solvers/ed_cnn.py` (not implemented)

**Internal Dependencies**: `core`, `lattice`

**External Dependencies**: `torch`, `numpy`, `scipy`

**Can Import Independently**: ✅ Yes (after core, lattice)

**Key Functions**:
- `diagonalize()` - Diagonalize Hamiltonian at each k-point (ED, IPT not implemented)

**Usage**:
```python
from condmatTensor.solvers import diagonalize
evals, evecs = diagonalize(Hk.tensor)
```

---

### LEVEL 4: Many-Body (Depends on LEVEL 1, LEVEL 2, LEVEL 3)

**Files**: `manybody/preprocessing.py`, `manybody/magnetic.py`, `manybody/dmft.py` (not implemented), `manybody/cdmft.py` (not implemented), `manybody/ipt.py` (not implemented), `manybody/ed.py` (not implemented)

**Internal Dependencies**: `core`, `lattice`, `solvers`, `manybody.preprocessing` (for magnetic.py)

**External Dependencies**: `torch`, `numpy`

**Can Import Independently**: ✅ Yes (after core, lattice, solvers)

**Key Classes**:
- `BareGreensFunction` - Non-interacting Green's function from Hamiltonian
- `SelfEnergy` - Self-energy container class
- `SpectralFunction` - Spectral function from Green's function
- `KondoLatticeSolver` - Kondo lattice model solver
- `SpinFermionModel` - Spin-fermion model

**Usage**:
```python
from condmatTensor.manybody.preprocessing import (
    generate_matsubara_frequencies,
    BareGreensFunction,
    SelfEnergy,
    SpectralFunction
)
from condmatTensor.manybody.magnetic import (
    KondoLatticeSolver,
    SpinFermionModel
)
```

---

### LEVEL 5: Analysis (Depends on LEVEL 1, LEVEL 2, LEVEL 3)

**Files**: `analysis/dos.py`, `analysis/bandstr.py`, `analysis/fermi.py` (not implemented), `analysis/qgt.py` (not implemented), `analysis/topology.py` (not implemented)

**Internal Dependencies**: `core`, `lattice`, `solvers.diag`, `analysis.qgt` (for topology)

**External Dependencies**: `torch`, `numpy`, `matplotlib`

**Can Import Independently**: ✅ Yes (after core, lattice, solvers)

**Key Classes**:
- `DOSCalculator` - Density of States calculator with Lorentzian broadening
- `ProjectedDOS` - Projected DOS extending DOSCalculator
- `BandStructure` - Band structure calculator and plotting

**Usage**:
```python
from condmatTensor.analysis import (
    DOSCalculator,
    ProjectedDOS,
    BandStructure
)
```

---

### LEVEL 6: Transport (Depends on LEVEL 1, LEVEL 2)

**Files**: `transport/rgf.py` (not implemented), `transport/transport.py` (not implemented)

**Internal Dependencies**: `core`, `lattice`

**External Dependencies**: `torch`, `numpy`, `scipy`

**Can Import Independently**: ✅ Yes (after core, lattice)

**Status**: ❌ Not started

---

### LEVEL 7: Optimization (Depends on LEVEL 4)

**Files**: `optimization/bayesian.py`, `optimization/magnetic.py`, `optimization/ml_interface.py` (not implemented)

**Internal Dependencies**: `manybody`

**External Dependencies**: `torch`, `numpy`, `botorch`/`scikit-optimize`/`sober-bo`

**Can Import Independently**: ✅ Yes (after manybody)

**Key Classes**:
- `BayesianOptimizer` - Bayesian optimization with multiple backends
- `MultiObjectiveOptimizer` - Multi-objective optimization
- `EffectiveArrayOptimizer` - Effective array downfolding optimizer

**Usage**:
```python
from condmatTensor.optimization import (
    BayesianOptimizer,
    MultiObjectiveOptimizer,
    EffectiveArrayOptimizer
)
```

---

### LEVEL 8: Interface (Depends on LEVEL 1, LEVEL 2)

**Files**: `interface/yaml_reader.py` (not implemented), `interface/wannier_reader.py` (not implemented)

**Internal Dependencies**: `core`, `lattice`

**External Dependencies**: `torch`, `numpy`, `pyyaml`

**Can Import Independently**: ✅ Yes (after core, lattice)

**Status**: ❌ Not started

---

### LEVEL 9: Logging (No Internal Dependencies)

**Files**: `logging/__init__.py` (not implemented)

**Internal Dependencies**: None

**External Dependencies**: None

**Can Import Independently**: ✅ Yes

**Status**: ❌ Not started

---

### LEVEL 10: Symmetry (Depends on LEVEL 1, LEVEL 2)

**Files**: `lattice/symmetry.py` (not implemented)

**Internal Dependencies**: `core`, `lattice`

**External Dependencies**: `torch`, `numpy`, `scipy` (2D self-implemented, 3D via spglib)

**Can Import Independently**: ✅ Yes (after core, lattice)

**Status**: ❌ Not started

---

## Import Chains for Common Workflows

This section shows actual import chains for common scientific workflows, demonstrating the dependency hierarchy in practice.

### Band Structure Calculation

```python
# LEVEL 1: Core
from condmatTensor.core import BaseTensor

# LEVEL 2: Lattice (uses LEVEL 1)
from condmatTensor.lattice import BravaisLattice, generate_k_path

# LEVEL 3: Solvers (uses LEVEL 1, 2)
from condmatTensor.solvers import diagonalize

# LEVEL 5: Analysis (uses LEVEL 1, 2, 3)
from condmatTensor.analysis import BandStructure
```

**Dependency Chain**: Core → Lattice → Solvers → Analysis

---

### DMFT Calculation

```python
# LEVEL 1: Core
from condmatTensor.core import BaseTensor

# LEVEL 2: Lattice (uses LEVEL 1)
from condmatTensor.lattice import BravaisLattice, generate_kmesh

# LEVEL 3: Solvers (uses LEVEL 1, 2)
from condmatTensor.solvers import diagonalize

# LEVEL 4: Many-Body preprocessing (uses LEVEL 1, 2)
from condmatTensor.manybody.preprocessing import (
    generate_matsubara_frequencies,
    BareGreensFunction,
    SelfEnergy
)

# LEVEL 4: Many-Body magnetic (uses LEVEL 1, 2, 3, preprocessing)
from condmatTensor.manybody.magnetic import (
    KondoLatticeSolver,
    SpinFermionModel
)
```

**Dependency Chain**: Core → Lattice → Solvers → Many-Body (preprocessing → magnetic)

---

### Bayesian Optimization

```python
# LEVEL 4: Many-Body (required for optimization target)
from condmatTensor.manybody.preprocessing import BareGreensFunction
from condmatTensor.manybody.magnetic import KondoLatticeSolver

# LEVEL 7: Optimization (uses LEVEL 4)
from condmatTensor.optimization import BayesianOptimizer
```

**Dependency Chain**: Core → Lattice → Solvers → Many-Body → Optimization

---

### Quantum Geometric Tensor & Topology

```python
# LEVEL 1: Core
from condmatTensor.core import BaseTensor

# LEVEL 2: Lattice
from condmatTensor.lattice import BravaisLattice, generate_kmesh

# LEVEL 3: Solvers
from condmatTensor.solvers import diagonalize

# LEVEL 5: Analysis - QGT (uses LEVEL 1)
from condmatTensor.analysis.qgt import hk_to_g_layer, compute_qgt

# LEVEL 5: Analysis - Topology (uses LEVEL 5 - QGT)
from condmatTensor.analysis.topology import chern_number, berry_curvature, ahe
```

**Dependency Chain**: Core → Lattice → Solvers → Analysis (QGT → Topology)

---

### Transport Calculations

```python
# LEVEL 1: Core
from condmatTensor.core import BaseTensor

# LEVEL 2: Lattice
from condmatTensor.lattice import BravaisLattice

# LEVEL 6: Transport (uses LEVEL 1, 2)
from condmatTensor.transport import RGF, Transport, stacking_rgf
```

**Dependency Chain**: Core → Lattice → Transport

---

## Circular Dependency Policy

**✅ NO CIRCULAR DEPENDENCIES** - The 10-level architecture ensures unidirectional dependencies:

### Core Policy Statement

- Lower levels (1-3) **never** depend on higher levels (4-10)
- Each level can **only** depend on levels with smaller numbers
- This is enforced by import order and module structure
- If a circular dependency is discovered, it must be resolved by refactoring

### Dependency Flow Diagram

```
LEVEL 1 (core) ──────────────────────────────────────┐
    ↓                                                │
LEVEL 2 (lattice) ──────┐                            │
    ↓                    │                            │
LEVEL 3 (solvers) ← LEVEL 8 (interface)              │
    ↓                    ↓                            │
LEVEL 4 (manybody) ←─────┘                            │
    ↓                                                 │
LEVEL 5 (analysis) ←─────────────────────────────────┤
    ↓                                                 │
LEVEL 7 (optimization) ←──────────────────────────────┘
    ↓
LEVEL 6 (transport) ← LEVEL 2 (lattice) ──────────────┘
```

### Example Valid Dependencies

| Module | Depends On | Valid? | Reason |
|--------|-----------|--------|--------|
| `lattice` | `core` | ✅ | LEVEL 2 depends on LEVEL 1 |
| `solvers` | `core`, `lattice` | ✅ | LEVEL 3 depends on LEVEL 1, 2 |
| `manybody` | `core`, `lattice`, `solvers` | ✅ | LEVEL 4 depends on LEVEL 1, 2, 3 |
| `analysis` | `core`, `lattice`, `solvers` | ✅ | LEVEL 5 depends on LEVEL 1, 2, 3 |
| `optimization` | `manybody` | ✅ | LEVEL 7 depends on LEVEL 4 |
| `transport` | `core`, `lattice` | ✅ | LEVEL 6 depends on LEVEL 1, 2 |
| `interface` | `core`, `lattice` | ✅ | LEVEL 8 depends on LEVEL 1, 2 |

### Example Invalid Dependencies (Would Be Circular)

| Invalid Dependency | Why Invalid? |
|-------------------|--------------|
| `core` depends on `lattice` | LEVEL 1 cannot depend on LEVEL 2 |
| `lattice` depends on `solvers` | LEVEL 2 cannot depend on LEVEL 3 |
| `solvers` depends on `manybody` | LEVEL 3 cannot depend on LEVEL 4 |
| `manybody` depends on `optimization` | LEVEL 4 cannot depend on LEVEL 7 |
| `analysis` depends on `optimization` | Would create circular dep |

### How to Resolve Circular Dependencies

If a circular dependency is discovered:

1. **Extract common code** to a lower level (create new module in LEVEL 1 or 2)
2. **Use dependency injection** (pass objects as parameters instead of importing)
3. **Split the module** into separate levels
4. **Use callbacks/events** (register functions instead of direct imports)

### Complete Import Dependency Matrix

| Level | Module Path | Status | Internal Dependencies | External Dependencies | Can Import Independently? |
|-------|-------------|--------|----------------------|----------------------|--------------------------|
| **1** | `condmatTensor.core.base` | ✅ | None | `torch` | ✅ Yes |
| **1** | `condmatTensor.core.device` | ✅ | None | `torch` | ✅ Yes |
| **1** | `condmatTensor.core.math` | ❌ | None | `torch`, `numpy`, `scipy` | ✅ Yes |
| **1** | `condmatTensor.core.gpu_utils` | ❌ | None | `torch` | ✅ Yes |
| **2** | `condmatTensor.lattice.model` | ✅ | `core.base`, `core.device` | `torch`, `numpy` | ✅ Yes (after core) |
| **2** | `condmatTensor.lattice.bzone` | ✅ | `core.device` | `torch`, `numpy` | ✅ Yes (after core) |
| **2** | `condmatTensor.lattice.symmetry` | ❌ | `core`, `lattice` | `torch`, `numpy`, `scipy` | ✅ Yes (after core, lattice) |
| **3** | `condmatTensor.solvers.diag` | ✅ | `core.base`, `lattice.model` | `torch`, `numpy`, `scipy` | ✅ Yes (after core, lattice) |
| **3** | `condmatTensor.solvers.ipt` | ❌ | `core` | `torch`, `numpy`, `scipy` | ✅ Yes (after core) |
| **3** | `condmatTensor.solvers.ed` | ❌ | `core` | `torch`, `numpy` | ✅ Yes (after core) |
| **3** | `condmatTensor.solvers.ed_cnn` | ❌ | `manybody.ed` | `torch`, `numpy` | ❌ No (needs manybody.ed) |
| **4** | `condmatTensor.manybody.preprocessing` | ✅ | `core`, `lattice` | `torch`, `numpy` | ✅ Yes (after core, lattice) |
| **4** | `condmatTensor.manybody.magnetic` | ✅ | `core`, `lattice`, `manybody.preprocessing` | `torch`, `numpy` | ✅ Yes (after core, lattice, preprocessing) |
| **4** | `condmatTensor.manybody.dmft` | ❌ | `core`, `lattice`, `manybody.ipt` | `torch`, `numpy` | ❌ No (needs ipt) |
| **4** | `condmatTensor.manybody.cdmft` | ❌ | `core`, `lattice`, `manybody.ipt`, `manybody.ed` | `torch`, `numpy` | ❌ No (needs ipt, ed) |
| **4** | `condmatTensor.manybody.ipt` | ❌ | `core` | `torch`, `numpy`, `scipy` | ✅ Yes (after core) |
| **4** | `condmatTensor.manybody.ed` | ❌ | `core` | `torch`, `numpy` | ✅ Yes (after core) |
| **5** | `condmatTensor.analysis.dos` | ✅ | `core`, `lattice`, `solvers.diag` | `torch`, `numpy`, `matplotlib` | ✅ Yes (after core, lattice, solvers) |
| **5** | `condmatTensor.analysis.bandstr` | ✅ | `core`, `lattice`, `solvers.diag` | `torch`, `numpy`, `matplotlib` | ✅ Yes (after core, lattice, solvers) |
| **5** | `condmatTensor.analysis.fermi` | ❌ | `core`, `lattice`, `solvers.diag` | `torch`, `numpy`, `matplotlib` | ✅ Yes (after core, lattice, solvers) |
| **5** | `condmatTensor.analysis.qgt` | ❌ | `core`, `core.math` | `torch`, `numpy` | ✅ Yes (after core) |
| **5** | `condmatTensor.analysis.topology` | ❌ | `analysis.qgt` | `torch`, `numpy`, `matplotlib` | ❌ No (needs qgt) |
| **6** | `condmatTensor.transport.rgf` | ❌ | `core`, `lattice` | `torch`, `numpy`, `scipy` | ✅ Yes (after core, lattice) |
| **6** | `condmatTensor.transport.transport` | ❌ | `transport.rgf` | `torch`, `numpy`, `matplotlib` | ❌ No (needs rgf) |
| **7** | `condmatTensor.optimization.bayesian` | ✅ | `manybody` | `torch`, `numpy`, `botorch`/`scikit-optimize` | ✅ Yes (after manybody) |
| **7** | `condmatTensor.optimization.magnetic` | ✅ | `manybody` | `torch`, `numpy`, `botorch`/`scikit-optimize` | ✅ Yes (after manybody) |
| **7** | `condmatTensor.optimization.ml_interface` | ❌ | `manybody` | `torch`, `numpy` | ✅ Yes (after manybody) |
| **8** | `condmatTensor.interface.yaml_reader` | ❌ | `core`, `lattice` | `torch`, `numpy`, `pyyaml` | ✅ Yes (after core, lattice) |
| **8** | `condmatTensor.interface.wannier_reader` | ❌ | `core`, `lattice` | `torch`, `numpy` | ✅ Yes (after core, lattice) |
| **9** | `condmatTensor.logging` | ❌ | None | - | ✅ Yes |
| **10** | `condmatTensor.lattice.symmetry` | ❌ | `core`, `lattice` | `torch`, `numpy`, `scipy`/`spglib` | ✅ Yes (after core, lattice) |

**Legend:**
- ✅ = Implemented
- ❌ = Not Implemented
- **Level**: Module level in the 10-level hierarchy
- **Can Import Independently?**: Whether the module can be imported after satisfying its internal dependencies

**Import Order Examples:**

```python
# Minimum import for basic usage (LEVEL 1-2)
from condmatTensor.core import BaseTensor, get_device
from condmatTensor.lattice import BravaisLattice, TightBindingModel

# Import for band structure (LEVEL 1-3, 5)
from condmatTensor.core import BaseTensor
from condmatTensor.lattice import BravaisLattice, generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure

# Import for DMFT (LEVEL 1-4)
from condmatTensor.core import BaseTensor, get_device
from condmatTensor.lattice import BravaisLattice, generate_kmesh
from condmatTensor.manybody.preprocessing import (
    generate_matsubara_frequencies, BareGreensFunction, SelfEnergy
)
from condmatTensor.manybody.magnetic import KondoLatticeSolver

# Import for Bayesian optimization (LEVEL 1-4, 7)
from condmatTensor.manybody import KondoLatticeSolver
from condmatTensor.optimization import BayesianOptimizer
```

**Key Dependency Rules:**
1. Import order must follow level numbers (lower levels first)
2. **No circular dependencies** - higher levels never import from lower levels
3. Module path format: `condmatTensor.<level>.<module>.<class/function>`
4. Use `__init__.py` exports for cleaner imports (e.g., `from condmatTensor.core import BaseTensor` instead of `from condmatTensor.core.base import BaseTensor`)

### Unit System and Coordinate Conventions

**Lattice Coordinates:**
- **User Input**: Fractional coordinates (dimensionless, in units of lattice vectors)
  - Example: `TightBindingModel.add_hopping("A", "B", [0, 0], 1.0)`
  - Displacements in fractional coordinates: `[0, 0]` = same unit cell, `[1, 0]` = next cell in x-direction
- **Internal Storage**: Cartesian coordinates
  - `BaseTensor.displacements` stores displacements in Cartesian (physical units)
  - Conversion: `R_cart = R_frac @ lattice.cell_vectors`
- **K-points**:
  - Input: Fractional (0 to 1) via `generate_kmesh(lattice, nk)`
  - Computation: Cartesian via `k_cart = k_frac @ reciprocal_vectors.T`

**Energy Units:**
- **Current Convention**: All energies in units of hopping parameter `t` (dimensionless)
  - Default: `t = -1.0` as energy scale reference
  - Example: Energy range `-4|t|` to `+4|t|` means `-4` to `+4` in internal units
  - Plot labels: `ylabel="Energy ($|t|$)"` (manually set)
- **No Explicit Unit Metadata**: Energy values stored as raw floats
  - Future: Optional `UnitSystem` dataclass for physical units (eV, meV)

**Wannier90 Compatibility (LEVEL 8):**
- Wannier90 hr.dat uses **Cartesian coordinates** (Angstrom or Bohr) and **eV** energies
- Required conversions:
  ```python
  # Coordinate: Cartesian → Fractional
  R_frac = R_cart @ torch.inverse(lattice_vectors)

  # Energy: eV → Hopping units
  H_reduced = H_eV / t_reference
  ```

examples/
├── kagome_bandstructure.py         ✅ Kagome: band structure, DOS (flat band validation)
├── kagome_f_spinful_bandstructure.py ✅ Kagome-F: 8 spinful bands with f-orbital weight plots
├── kagome_spinful_bandstructure.py   ✅ Spinful Kagome: Zeeman splitting, local magnetic moments
├── kagome_spinful_with_b_field.py    ✅ Zeeman effect: B-field sweeps, in-plane vs out-of-plane
├── kagome_with_f_bandstructure.py    ✅ Kagome-F: 4-site model, DOS, spectral functions
├── kagome_f_effective_array.py      ✅ Effective array optimizer: 8→6 band downfolding
├── kagome_f_spinful_effective.py    ✅ NEW: Spinful effective model with proper lattice handling (spinless → spinful conversion)
├── test_magnetic_additional.py        ✅ Tests: Pauli matrices, self-consistency, Kondo solver, Bayesian backends
├── test_bayesian_backends.py         ✅ Backend comparison: SOBER, BoTorch, Simple
├── test_sober_backend_specific.py     ✅ SOBER-specific: bounds format, device handling, edge cases
├── gpu_performance_benchmark.py     ✅ CPU vs GPU performance for all backends
├── bayesian_parameter_sweep_example.py ✅ Grid search vs Bayesian optimization comparison
├── phase1_kagome_basic.py           # Kagome: band structure, DOS (flat band) [LEGACY - replaced by kagome_bandstructure.py]
├── phase2_kagome_dmft.py            # Kagome Hubbard model with DMFT
├── phase3_kagome_qgt.py             # Kagome QGT, Chern number, AHE
├── phase4_kagome_transport.py      # Kagome lattice transport
├── phase4_heterostructure.py      # Au-MoSe2-Graphene stacking
├── phase5_kagome_optimize.py       # Optimize U, beta for Kagome
├── parallel_ed_cnn.py              # Compare ED vs CNN-CI on Kagome cluster
└── phase10_symmetry_reduction.py  # Memory-efficient IBZ calculations (after LEVEL 8)
```

**Note**: All examples use generic building blocks (`BravaisLattice`, `BaseTensor`) to construct specific lattices like Kagome.

## Current Implementation: Detailed Method Reference

This section documents all implemented classes and their methods as of 2026-01-22.

### LEVEL 1: Core Module (`src/condmatTensor/core/`)

#### `BaseTensor` class (`base.py` - 138 lines)

**Purpose**: Unified tensor representation for all physics objects (Hamiltonians, Green's functions, Self-energies).

**Attributes**:
- `tensor: torch.Tensor` - Underlying tensor data
- `labels: List[str]` - Semantic labels for each dimension (e.g., `['k', 'orb_i', 'orb_j']`)
- `orbital_names: List[str] | None` - Physical orbital names (e.g., `['px', 'py', 'pz']`)
- `displacements: torch.Tensor | None` - Real-space displacement vectors for H(R), shape (N_R, dim)

**Methods**:
| Method | Description |
|--------|-------------|
| `__init__(tensor, labels, orbital_names=None, displacements=None)` | Initialize with validation (len(labels) == tensor.ndim) |
| `to_k_space(k)` | Fourier transform from real-space (R) to momentum-space (k). H(k) = Σ_R H(R)·exp(i·k·R). Uses einsum for flexible tensor contraction. Returns BaseTensor with 'R' replaced by 'k'. |
| `to(device)` | Move tensor to device (CPU/GPU). Returns new BaseTensor. |
| `shape` (property) | Return tensor shape |
| `ndim` (property) | Return number of dimensions |
| `dtype` (property) | Return tensor dtype |
| `__repr__()` | String representation: `BaseTensor(shape=..., labels=..., dtype=...)` |

**Not Yet Implemented in LEVEL 1**:
- `core/math.py` - Tensor math utilities, Berry curvature helpers
- `core/gpu_utils.py` - Device selection, memory estimates, chunking

---

### LEVEL 2: Lattice Module (`src/condmatTensor/lattice/`)

#### `BravaisLattice` class (`model.py` - lines 185-291)

**Purpose**: Bravais lattice with multiple sites per unit cell.

**Attributes**:
- `cell_vectors: torch.Tensor` - Lattice vectors, shape (dim, dim)
- `basis_positions: List[torch.Tensor]` - Basis atom positions (fractional coordinates)
- `num_orbitals: int` - Number of orbitals per site
- `dim: int` - Spatial dimension (2 or 3)
- `num_sites: int` - Number of basis sites in unit cell (computed)

**Properties**:
| Property | Returns |
|----------|---------|
| `num_basis` | Number of basis sites in unit cell |
| `total_orbitals` | Total orbitals in unit cell (num_sites × num_orbitals) |

**Methods**:
| Method | Description |
|--------|-------------|
| `__init__(cell_vectors, basis_positions, num_orbitals=1)` | Initialize lattice |
| `reciprocal_vectors()` | Compute reciprocal lattice vectors. 2D: b_i = 2π·ε_ij·a_j/\|a₁×a₂\|. 3D: b_i = 2π·ε_ijk·a_j×a_k/(a₁·(a₂×a₃)) |
| `high_symmetry_points()` | Return high-symmetry points for triangular lattice: G=(0,0), K=(1/3,1/√3), M=(1/2,0) |
| `__repr__()` | String: `BravaisLattice(dim=..., num_sites=..., num_orbitals=...)` |

---

#### `TightBindingModel` class (`model.py` - lines 7-182)

**Purpose**: General tight-binding model builder with symbolic hopping terms.

**Attributes**:
- `lattice: BravaisLattice` - Lattice structure
- `hoppings: List[Tuple]` - List of (orb_i, orb_j, displacement, value) tuples
- `orbital_labels: List[str]` - Orbital names (e.g., ['A', 'B', 'C'])
- `_label_to_idx: Dict[str, int]` - Mapping from orbital labels to indices (internal)

**Methods**:
| Method | Description |
|--------|-------------|
| `__init__(lattice, orbital_labels=None, hoppings=None)` | Initialize model. Creates default labels if none provided. |
| `_resolve_orbital(orb)` | Convert orbital label or index to integer index (internal) |
| `add_hopping(orb_i, orb_j, displacement, value=1.0, add_hermitian=True)` | Add hopping term. Supports both integer indices and string labels. If add_hermitian=True, automatically adds Hermitian conjugate (swap orbitals, negate displacement, conjugate value). |
| `build_HR()` | Build real-space Hamiltonian H(R) from hopping terms. Returns BaseTensor with labels=['R', 'orb_i', 'orb_j'], orbital_names set, displacements in Cartesian coordinates. |
| `build_Hk(k_path)` | Build k-space Hamiltonian H(k) directly from hopping terms. Returns BaseTensor with labels=['k', 'orb_i', 'orb_j'], orbital_names set. H(k) = Σ_R H(R)·exp(i·k·R) |

---

#### B-Zone Functions (`bzone.py` - 94 lines)

| Function | Description |
|----------|-------------|
| `generate_kmesh(lattice, nk, device=None)` | Generate uniform k-mesh in fractional coordinates. Returns shape (nk^dim, dim). Uses meshgrid with indexing='ij'. |
| `generate_k_path(lattice, points, n_per_segment, device=None)` | Generate k-path along high-symmetry lines. Returns (k_path, ticks) where ticks is List[Tuple[int, str]] for plot markers. Interpolates between high-symmetry points. |
| `k_frac_to_cart(k_frac, lattice)` | Convert k-points from fractional to Cartesian: k_cart = k_frac @ b where b are reciprocal vectors |

---

### LEVEL 3: Solvers Module (`src/condmatTensor/solvers/`)

#### `diagonalize` function (`diag.py` - 36 lines)

**Purpose**: Diagonalize Hamiltonian at each k-point. H_k\|ψₙk⟩ = εₙk\|ψₙk⟩

**Signature**:
```python
def diagonalize(Hk: torch.Tensor, hermitian: bool = True) -> tuple[torch.Tensor, torch.Tensor]
```

**Parameters**:
- `Hk`: Hamiltonian in k-space, shape (N_k, N_orb, N_orb)
- `hermitian`: If True, use torch.linalg.eigh (faster, assumes Hermitian). If False, use torch.linalg.eig (general).

**Returns**:
- `eigenvalues`: Eigenvalues εₙk, shape (N_k, N_orb)
- `eigenvectors`: Eigenvectors, shape (N_k, N_orb, N_orb) - column n corresponds to εₙk

**Not Yet Implemented in LEVEL 3**:
- `solvers/ipt.py` - IPT impurity solver
- `solvers/ed.py` - Exact diagonalization
- `solvers/ed_cnn.py` - CNN-CI ED solver

---

### LEVEL 5: Analysis Module (`src/condmatTensor/analysis/`)

#### `DOSCalculator` class (`dos.py` - lines 20-130)

**Purpose**: Density of States calculator with Lorentzian broadening.

**Formula**: ρ(ω) = (1/N_k) Σ_{k,n} (η/π) / [(ω - εₙ(k))² + η²]

**Attributes**:
- `omega: torch.Tensor | None` - Energy grid
- `rho: torch.Tensor | None` - DOS values
- `eta: float | None` - Broadening width

**Methods**:
| Method | Description |
|--------|-------------|
| `__init__()` | Initialize calculator (all attributes None) |
| `from_eigenvalues(E_k, omega, eta=0.02)` | Compute DOS from eigenvalues using Lorentzian broadening. Vectorized computation. Returns (omega, rho) tuple and stores in self. |
| `plot(ax, energy_range, ylabel, xlabel, title, fontsize, fill, **kwargs)` | Plot stored DOS results. If ax=None, creates new figure. Optional fill under curve. |

---

#### `ProjectedDOS` class (`dos.py` - lines 133-279)

**Purpose**: Projected Density of States (PDOS) extending DOSCalculator.

**Additional Attributes**:
- `pdos: torch.Tensor | None` - Shape (n_omega, N_orb)
- `orbital_labels: list[str] | None` - Orbital names

**Methods**:
| Method | Description |
|--------|-------------|
| `__init__()` | Initialize (inherits from DOSCalculator) |
| `from_eigenvalues(E_k, U, omega, eta=0.02, orbital_labels=None)` | Compute PDOS from eigenvalues and eigenvectors. Input U shape (N_k, N_band, N_orb). Computes |ψₙk(i)|² weights. Returns (omega, rho) for total DOS, stores PDOS in self.pdos. |
| `get_projected_dos()` | Return PDOS tensor, shape (n_omega, N_orb) |
| `plot_projected(ax, energy_range, ylabel, xlabel, title, fontsize, stacked, **kwargs)` | Plot PDOS. Stacked or overlaid orbital contributions. Automatic legend with orbital labels. |

---

#### `BandStructure` class (`bandstr.py` - 237 lines)

**Purpose**: Band structure calculator and plotting.

**Attributes**:
- `k_path: torch.Tensor | None` - K-points along path
- `eigenvalues: torch.Tensor | None` - Band energies
- `ticks: List[Tuple[int, str]] | None` - High-symmetry point markers

**Methods**:
| Method | Description |
|--------|-------------|
| `__init__()` | Initialize calculator (all attributes None) |
| `compute(eigenvalues, k_path, ticks=None)` | Store band structure results |
| `plot(ax, energy_range, ylabel, title, fontsize, **kwargs)` | Plot band structure. Vertical lines at high-symmetry points. Grid overlay. |
| `plot_with_dos(eigenvalues_mesh, omega, eta=0.02, energy_range, ylabel, dos_xlabel, title, fontsize, figsize, dos_color, **kwargs)` | Combined plot: left panel = band structure along k-path, right panel = DOS from full k-mesh (horizontal orientation). Shared y-axis (energy). |

**Not Yet Implemented in LEVEL 5**:
- `analysis/qgt.py` - Quantum Geometric Tensor functions
- `analysis/topology.py` - Berry curvature, Chern number, Z₂ invariant
- `analysis/fermi.py` - FermiSurface class

---

## Module Formalism and Equations

This section provides the mathematical formalism for each module, verified against current condensed matter physics literature.

### LEVEL 1: Core Module Formalism

**Fourier Transform: R → k**

The tight-binding Fourier transform from real-space to k-space:

```
H_{αβ}(k) = Σ_R H_{αβ}(R) · exp(i·k·R)
```

where:
- `α, β` are orbital indices
- `R` is the lattice vector (displacement)
- `k` is the momentum vector

**Tensor Implementation:**
```python
phases = exp(i * k @ R.T)      # (N_k, N_R)
Hk = einsum('ij,jkl->ikl', phases, Hr)  # (N_k, N_orb, N_orb)
```

### LEVEL 2: Lattice Module Formalism

**Bravais Lattice Definition:**
```
L = {R = n₁a₁ + n₂a₂ + n₃a₃ | nᵢ ∈ ℤ}
```

**Reciprocal Lattice:**
```
b₁ = 2π (a₂ × a₃) / (a₁ · (a₂ × a₃))
b₂ = 2π (a₃ × a₁) / (a₁ · (a₂ × a₃))
b₃ = 2π (a₁ × a₂) / (a₁ · (a₂ × a₃))
```

**Orthogonality:**
```
aᵢ · bⱼ = 2π δᵢⱼ
```

### LEVEL 10: Symmetry Module Formalism

**Purpose:** Reduce k-mesh size and memory usage by exploiting lattice symmetries.

**Symmetry Detection Strategy:**
- **2D lattices**: Self-implemented method (analytic point group detection from cell_vectors)
- **3D lattices**: spglib library (robust space group detection)

**Point Group Operations:**
```
R(α, θ) · k = k'
```
where:
- `R(α, θ)` is a rotation by angle θ about axis α
- `k'` is the symmetry-equivalent k-point

**Common 2D Point Groups (Self-Implemented):**
```
C₂: 180° rotation (rectangular, centered rectangular)
C₃: 120° rotation (triangular, hexagonal)
C₄: 90° rotation (square)
C₆: 60° rotation (hexagonal)

D₂: C₂ + 2 reflections
D₃: C₃ + 3 reflections
D₄: C₄ + 4 reflections
D₆: C₆ + 6 reflections
```

**2D Point Group Detection Algorithm:**
```python
def detect_2d_point_group(cell_vectors: torch.Tensor) -> str:
    """Detect 2D point group from Bravais lattice cell vectors.

    Algorithm:
    1. Compute lattice vectors a1, a2 and angle between them
    2. Check rotational symmetries by testing R(θ)·cell_vectors ≈ cell_vectors
    3. Check reflection symmetries about axes
    4. Return point group name (e.g., 'C4v', 'D6h')
    """
    # Analytic detection from cell_vectors shape
    # - Square: |a1| = |a2|, angle = 90° → C4v
    # - Hexagonal: |a1| = |a2|, angle = 60° → D6h
    # - Triangular: |a1| = |a2|, angle = 60° → D3h
    # - Rectangular: |a1| ≠ |a2|, angle = 90° → C2v
    # - Centered rectangular: special case → C2v
    ...
```

**3D Space Group Detection (spglib):**
```python
import spglib

def detect_3d_symmetry(lattice: BravaisLattice,
                       positions: torch.Tensor,
                       numbers: list[int]) -> dict:
    """Detect 3D space group using spglib.

    Returns:
        dict with spacegroup number, international symbol, operations
    """
    cell = (lattice.cell_vectors.cpu().numpy(),
            positions.cpu().numpy(),
            numbers)
    symmetry = spglib.get_symmetry(cell)
    dataset = spglib.get_spacegroup_type(symmetry['spacegroup'])
    return dataset
```

**Time-Reversal Symmetry:**
```
Θ · H(k) · Θ⁻¹ = H(-k)
```
For spinless systems with TRS:
```
εₙ(k) = εₙ(-k)
```

**Irreducible Brillouin Zone (IBZ):**
```
IBZ = {k ∈ BZ | k ≺ g(k) for all g ∈ G}
```
where `≺` is a canonical ordering and `G` is the symmetry group.

**Symmetry-Weighted Sum:**
```
(1/N_k) Σ_{k∈BZ} f(k) = Σ_{k∈IBZ} w_k · f(k)
```
where `w_k = n_k / N_k` is the symmetry weight (`n_k` is the number of equivalent k-points).

**Memory Savings:**
```
Memory_reduction = |BZ| / |IBZ|
```
Example: Square lattice with C₄v symmetry → 8x reduction
Example: Hexagonal lattice with D₆h symmetry → 12x reduction

**Symmetry Reduction Algorithm:**
```
2D:
1. Detect point group from BravaisLattice.cell_vectors (self-implemented)
2. Generate full k-mesh in BZ
3. Apply symmetry operations to find equivalent k-points
4. Select unique k-points (IBZ) with weights
5. Compute observables on IBZ only
6. Apply weights when summing/integrating

3D:
1. Use spglib to get space group symmetry operations
2. Generate full k-mesh in BZ
3. Apply spglib symmetry operations to find equivalent k-points
4. Select unique k-points (IBZ) with weights
5. Compute observables on IBZ only
6. Apply weights when summing/integrating
```

**Tensor Labels:**
- `k_ibz` → labels=['k', 'dim'] with `weights: Tensor[N_k]`
- Symmetry weights stored as attribute: `k_ibz.weights`

### LEVEL 3: Solvers Module Formalism

**Diagonalization:**
```
H(k) |uₙ(k)⟩ = εₙ(k) |uₙ(k)⟩
```

**IPT (Iterated Perturbation Theory) Self-Energy:**
```
Σ(iωₙ) = U² · χ₀(iωₙ) · G(iωₙ)
```

where the particle-hole bubble is:
```
χ₀(iωₙ) = (T/N_k) Σ_{k,νₘ} G(k, iνₘ) G(k, iνₘ + iωₙ)
```

### LEVEL 4: Many-Body Module Formalism

**Green's Function with Self-Energy:**
```
G(k, iωₙ) = [iωₙ + μ - H(k) - Σ(iωₙ)]^(-1)
```

**Tensor Labels:**
- `G(k, iωₙ)` → labels=['k', 'orb_i', 'orb_j', 'iwn']
- `Σ(iωₙ)` → labels=['iwn'] (pure imaginary)

**Matsubara Frequencies (Fermionic):**
```
iωₙ = iπ(2n + 1)/β,  n ∈ ℤ
```
where `β = 1/(k_B T)` is the inverse temperature.

**Tensor Label:** `'iwn'` (pure imaginary, dtype=torch.complex128)

**DMFT Self-Consistency Loop (SingleSiteDMFT):**

```
┌─────────────────────────────────────────────────────────────┐
│  1. Start with guess Σ(iωₙ)                                │
│  2. Compute lattice G(k,iω) = [iω+μ-H(k)-Σ(iω)]^(-1)     │
│  3. Extract local G_loc(iω) = (1/N_k) Σ_k G(k,iω)          │
│  4. Compute Weiss field: G₀^(-1)(iω) = G_loc^(-1)(iω) + Σ(iω) │
│  5. Solve impurity: G_imp(iω) = 𝒢[G₀]                     │
│  6. Extract new Σ(iω) = G₀^(-1)(iω) - G_imp^(-1)(iω)      │
│  7. Check convergence, iterate if needed                  │
└─────────────────────────────────────────────────────────────┘
```

**Dyson Equation:**
```
G^(-1) = G₀^(-1) - Σ
```

### LEVEL 5: Analysis Module Formalism

**DOS from Eigenvalues:**
```
ρ(ω) = (1/N_k) Σ_{k,n} (η/π) / [(ω - εₙ(k))² + η²]
```

**DOS from Green's Function (Future Work):**
```
A(k, ω) = -(1/π) Im[Gᴿ(k, ω)]
ρ(ω) = (1/N_k) Σ_k A(k, ω)
```
where `Gᴿ(k, ω) = [ω + μ + iη - H(k) - Σᴿ(ω)]^(-1)` is the retarded Green's function.

**Quantum Geometric Tensor (QGT):**
```
Q_{mn}^{μν}(k) = ⟨∂_kμ u_m(k) | ∂_kν u_n(k)⟩
```

**Berry Curvature (from QGT):**
```
Ω_n^μν(k) = 2 · Im[Q_{nn}^{μν}(k)]
```

For 2D systems (xy component):
```
Ω_n(k) = 2 · Im[Q_{nn}^{xy}(k)]
```

**Chern Number:**
```
C = (1/2π) ∫_BZ d²k Ω(k)
```

**Discrete Form (Fukui-Hatsugai-Suzuki):**
```
C = (1/2π) Σ_{k∈BZ} F(k)
```

where:
```
F(k) = Im[ln(⟨u(k)|u(k+Δ_x)⟩ · ⟨u(k+Δ_x)|u(k+Δ_x+Δ_y)⟩
                    · ⟨u(k+Δ_x+Δ_y)|u(k+Δ_y)⟩ · ⟨u(k+Δ_y)|u(k)⟩)]
```

**Anomalous Hall Conductivity (Kubo Formula):**
```
σ_xy = -(e²/ℏ) ∫ [d²k/(2π)²] Σ_n f(εₙ(k)) Ωₙ(k)
```

At zero temperature:
```
σ_xy = -(e²/ℏ) Σ_{n: εₙ < μ} Cₙ
```

**Z₂ Invariant (Time-Reversal Symmetric Systems):**

For systems with time-reversal symmetry and spin-orbit coupling, the Z₂ invariant distinguishes trivial from topological insulators.

**Time-Reversal Polarization (Fu-Kane Formula):**
```
Z₂ = (1/2π) [∑_{i=1}^{4} θ_i - ∑_{i=1}^{4} θ_i^ref] mod 2
```

where θᵢ are the Berry phases along time-reversal invariant loops (TRIM).

**Parity at TRIM Points:**
```
δ_i = ∏_{m=1}^{N_occ} ξ_m(Λ_i)
```

where ξ_m(Λi) is the parity eigenvalue of the m-th occupied band at time-reversal invariant momentum Λᵢ.

**Strong Z₂ Index:**
```
ν₀ = (∏_{i=1}^{8} δ_i)^{1/2} ∈ {0, 1}
```

ν₀ = 1: Strong topological insulator
ν₀ = 0: Trivial or weak topological insulator

**For 2D Systems:**
```
Z₂ = C mod 2
```

where C is the Chern number (Kane-Mele model).

### LEVEL 6: Transport Module Formalism

**Landauer-Büttiker Transmission:**
```
T(E) = Tr[Γ_L(E) G(E) Γ_R(E) G†(E)]
```

where:
- `G(E) = (E - H + iη)^(-1)` is the retarded Green's function
- `Γ_{L,R}(E) = i[Σ_{L,R}(E) - Σ_{L,R}†(E)]` are coupling functions

**Two-Terminal Conductance:**
```
G = (e²/h) T(E_F)
```

**RGF Algorithm (Forward Sweep):**
```
g₁ = (E - H₁ - Σ_L)^(-1)
g_i = (E - H_i - V_{i,i-1} g_{i-1} V_{i-1,i})^(-1)
```

### LEVEL 7: Optimization Module

Uses standard Bayesian optimization (scikit-optimize) - no physics-specific formalism.

### LEVEL 8: Interface Module

**Wannier90 hr.dat Format:**
```
H(R) format: (R_x, R_y, R_z), orbital_i, orbital_j, Re[H], Im[H]
```

---

## Phase-by-Phase Implementation

The following phases align with the example file naming scheme. Each phase builds upon the previous ones.

### Phase 1: Core Infrastructure with Unified BaseTensor

**Corresponding Example**: `examples/phase1_kagome_basic.py`

**Goal**: Unified tensor infrastructure with lattice and input interfaces.

**Key Components**:
- `BaseTensor` class with tensor + labels + orbital names
- GPU utilities (device selection, chunking)
- Math utilities (tensor operations)
- Lattice models (BravaisLattice, ClusterLattice)
- BZ meshes and paths
- YAML and Wannier90 input interfaces

**Example**: 2D square lattice from YAML
```python
# lattice.yaml
lattice:
  dim: 2
  cell_vectors: [[1.0, 0.0], [0.0, 1.0]]
  basis_positions: [[0.0, 0.0]]
  num_orbitals: 1

hoppings:
  - {site_i: 0, site_j: 0, delta: [1, 0], orbital_i: 0, orbital_j: 0, value: -1.0}
  - {site_i: 0, site_j: 0, delta: [0, 1], orbital_i: 0, orbital_j: 0, value: -1.0}
```

```python
from condmatTensor.interface.yaml_reader import YAMLReader
reader = YAMLReader('lattice.yaml')
lattice = reader.build_lattice()
Hr = reader.build_hamiltonian()  # BaseTensor with labels=['R', 'orb_i', 'orb_j']
Hk = Hr.to_k_space(k_mesh)      # BaseTensor with labels=['k', 'orb_i', 'orb_j']
```

**Example: Kagome lattice from generic building blocks** (see `examples/phase1_kagome_basic.py`)
```python
# Build Kagome lattice using generic BravaisLattice
# Kagome: triangular lattice with 3 sites per unit cell
a = 1.0
cell_vectors = torch.tensor([[a, 0.0], [a/2, a*sqrt(3)/2]])
basis_positions = [
    torch.tensor([0.0, 0.0]),
    torch.tensor([a/2, 0.0]),
    torch.tensor([a/4, a*sqrt(3)/4])
]

# Define nearest-neighbor hoppings for Kagome
displacements = torch.tensor([
    [0, 0], [1, 0], [0, 1], [-1, 1],  # intra-unit cell
    [1, 0], [0, 1], [-1, 1]           # inter-unit cell
])
hopping_tensor = torch.zeros((len(displacements), 3, 3), dtype=torch.complex128)
# ... fill hopping_tensor with Kagome hopping pattern (t=-1.0) ...

lattice = BravaisLattice(cell_vectors, basis_positions, num_orbitals=3)
Hr = BaseTensor(hopping_tensor, labels=['R', 'orb_i', 'orb_j'])
Hr.displacements = displacements
Hk = Hr.to_k_space(k_mesh)  # Fourier transform to k-space
```

### Phase 2: Analysis Modules with Integrated Plotting

**Corresponding Example**: `examples/phase1_kagome_basic.py` (same as Phase 1 - analysis included)

**Goal**: Basic analysis (DOS, band structure, Fermi surface) with plotting.

**Key Components**:
- `analysis/dos.py` with `DOSCalculator` class
- `analysis/bandstr.py` with `BandStructure` class
- `analysis/fermi.py` with `FermiSurface` class
- `solvers/diag.py` for diagonalization

**Example**: Kagome lattice band structure and DOS (see `examples/phase1_kagome_basic.py`)
```python
from condmatTensor.analysis import DOSCalculator, BandStructure
from condmatTensor.solvers import diagonalize
from condmatTensor.lattice import BravaisLattice, generate_k_path, generate_kmesh
from condmatTensor.core import BaseTensor
import torch

# Build Kagome lattice (as shown in Phase 1 example)
lattice, Hr = build_kagome_lattice()  # Using generic building blocks

# Band structure along high-symmetry path
k_path = generate_k_path(lattice, ['G', 'K', 'M', 'G'], 100)
Hk_path = Hr.to_k_space(k_path)
E, U = diagonalize(Hk_path)  # E: (N_k, N_band=3), U: (N_k, N_band, N_orb=3)

# Plot band structure
bs = BandStructure()
bs.compute(k_path, E)  # Store results internally
bs.plot(labels=['G', 'K', 'M', 'G'])

# DOS - compute and plot using same instance
k_mesh = generate_kmesh(lattice, 50)
Hk_mesh = Hr.to_k_space(k_mesh)
E_mesh = diagonalize(Hk_mesh)[0]
omega = torch.linspace(-4, 4, 500)
dos = DOSCalculator()
dos.from_eigenvalues(E_mesh, omega, eta=0.05)  # Stores in dos.omega, dos.rho
dos.plot()  # Uses stored results
```

### Phase 3: DMFT Loop with IPT

**Corresponding Example**: `examples/phase2_kagome_dmft.py`

**Goal**: Full DMFT self-consistency with preprocessing utilities.

**Key Components**:
- `dmft/preprocessing.py` for preprocessing
- `solvers/ipt.py` for IPT solver
- `dmft/singlesite.py` for `SingleSiteDMFTLoop` class

**Workflow**:
```
H(k) + Σ(ω) → G(k,ω) → G_loc(ω) → G0(ω) → IPT → Σ_new(ω)
```

**Example**: Kagome Hubbard model with DMFT (see `examples/phase2_kagome_dmft.py`)
```python
# Import DMFT components
from condmatTensor.manybody import DMFTLoop
from condmatTensor.manybody.preprocessing import generate_matsubara_frequencies
from condmatTensor.manybody import IPTSolver
from condmatTensor.lattice import BravaisLattice, generate_kmesh
from condmatTensor.core import BaseTensor
import torch

# Build Kagome Hubbard model
# Use centered Kagome: 4 sites per unit cell, Hubbard U only on center site
lattice, Hr = build_centered_kagome_lattice(U=4.0)  # Using generic building blocks

# Setup DMFT
k_mesh = generate_kmesh(lattice, 32)
omega = generate_matsubara_frequencies(beta=10.0, n_freq=256)
solver = IPTSolver(U=4.0, beta=10.0)

# DMFT loop
Hk_mesh = Hr.to_k_space(k_mesh)
dmft = SingleSiteDMFTLoop(Hk_mesh, omega, solver, mu=0.0)
Sigma = dmft.run()

# Plot spectral function
from condmatTensor.analysis import DOSCalculator
dos = DOSCalculator()
omega_real = torch.linspace(-4, 4, 500)
A = dos.spectral_function(Sigma, omega_real, eta=0.05)
dos.plot(omega_real, A)

# Verify: H(k) -> G_loc -> DMFT self-consistency
# Check convergence of self-energy and local Green's function
```

### Phase 4: QGT and Topology with Plugin Architecture

**Corresponding Examples**:
- `examples/phase3_kagome_qgt.py` - Kagome QGT, Chern number, AHE
- `examples/phase4_kagome_transport.py` - Kagome lattice transport
- `examples/phase4_heterostructure.py` - Au-MoSe2-Graphene stacking

**Goal**: Quantum Geometric Tensor with dual input methods + Transport properties.

**Key Components**:
- `analysis/qgt.py` with plugin methods (`hk_to_g_layer`, `hr_to_g_layer`, `compute_qgt`)
- `analysis/topology.py` for Chern/Z2/AHE functions
- `transport/rgf.py` for `RGF` class and `stacking_rgf` function
- `transport/transport.py` for `Transport` class

**Plugin Architecture**:
```python
# From H(k) using autograd
from condmatTensor.analysis.qgt import hk_to_g_layer, compute_qgt
g_layer = hk_to_g_layer(Hk, k_mesh)
qgt = compute_qgt(g_layer)

# From H(R) using analytic derivatives
from condmatTensor.analysis.qgt import hr_to_g_layer, compute_qgt
g_layer = hr_to_g_layer(Hr, displacements, k_mesh)
qgt = compute_qgt(g_layer)
```

**Example**: Kagome lattice QGT, Chern number, AHE (see `examples/phase3_kagome_qgt.py`)
```python
from condmatTensor.analysis.qgt import hk_to_g_layer, compute_qgt
from condmatTensor.analysis.topology import chern_number, berry_curvature, ahe
from condmatTensor.analysis.topology import TopologyAnalysis
from condmatTensor.lattice import BravaisLattice, generate_kmesh
from condmatTensor.solvers import diagonalize
from condmatTensor.core import BaseTensor
import torch

# Build Kagome lattice with spin-orbit coupling (for non-trivial topology)
lattice, Hr = build_kagome_with_soc(lambda_so=0.1)  # Using generic building blocks
k_mesh = generate_kmesh(lattice, 64)
Hk = Hr.to_k_space(k_mesh)

# Compute QGT from H(k) using autograd
g_layer = hk_to_g_layer(Hk, k_mesh)
Q = compute_qgt(g_layer)
# Q shape: (N_k, N_band, N_band, dim, dim)
# labels: ['k', 'band_i', 'band_j', 'dim_i', 'dim_j']

# Chern number (integrates Berry curvature from QGT)
# chern_number() extracts imaginary part of Q and integrates over BZ
C = chern_number(Q, k_mesh, n_bands=Hk.tensor.shape[1])
print(f"Chern number: {C}")

# Berry curvature distribution
E, U = diagonalize(Hk)
Omega = berry_curvature(Q)  # Shape: (N_k, N_band), labels: ['k', 'band']

# Plot Berry curvature
topo = TopologyAnalysis()
topo.compute_berry_curvature(Q)  # Stores internally
topo.plot(k_mesh)

# Anomalous Hall conductivity
sigma_xy = ahe(Q, E, mu=0.0, k_mesh=k_mesh)
topo.compute_ahe(Q, E, mu=0.0, k_mesh=k_mesh)
topo.plot_ahe()
```

**Example: Haldane model** (alternative example)
```python
# Build Haldane model (complex hopping)
Hr_haldane = build_haldane_model(t1=1.0, t2=0.2, phi=pi/2)
Hk_haldane = Hr_haldane.to_k_space(k_mesh)

# Compute QGT and Chern
from condmatTensor.analysis.qgt import hk_to_g_layer, compute_qgt
from condmatTensor.analysis.topology import chern_number, ahe
g_layer = hk_to_g_layer(Hk_haldane, k_mesh)
Q = compute_qgt(g_layer)  # Shape: (N_k, N_band, N_band, dim, dim)
E_haldane = diagonalize(Hk_haldane)[0]
C = chern_number(Q, k_mesh, n_bands=Hk_haldane.tensor.shape[1])

# AHE
sigma_xy = ahe(Q, E_haldane, mu=0.0, k_mesh=k_mesh)
```

### Phase 5: ML/Bayesian Optimization

**Corresponding Example**: `examples/phase5_kagome_optimize.py`

**Goal**: Automated parameter search.

**Key Components**:
- `optimization/bayesian.py` for `BayesianOptimizer` class
- `optimization/ml_interface.py` for ML surrogate models

**Example**: Optimize U, beta for Kagome Hubbard model (see `examples/phase5_kagome_optimize.py`)
```python
from condmatTensor.optimization.bayesian import BayesianOptimizer
from condmatTensor.manybody import DMFTLoop
from condmatTensor.manybody.preprocessing import generate_matsubara_frequencies
from condmatTensor.manybody import IPTSolver
from condmatTensor.lattice import BravaisLattice, generate_kmesh
from condmatTensor.core import BaseTensor
import torch

# Build Kagome Hubbard model
lattice, Hr = build_centered_kagome_lattice()  # Using generic building blocks
Hk = Hr.to_k_space(k_mesh)
omega = generate_matsubara_frequencies(beta=10.0, n_freq=256)

def objective(params):
    U, beta = params
    solver = IPTSolver(U=U, beta=beta)
    dmft = SingleSiteDMFTLoop(Hk, omega, solver, mu=0.0)
    Sigma = dmft.run()
    # Minimize spectral weight at Fermi level (Mott transition indicator)
    return -Sigma.imag.max()

# Search for Mott transition
opt = BayesianOptimizer(objective, bounds=[(0, 10), (1, 50)])
result = opt.optimize(n_iter=50)
print(f"Optimal U={result.x[0]:.2f}, beta={result.x[1]:.2f}")
```

### Phase 4 Transport: RGF Design

**Formalism:**

**Landauer-Büttiker Formula:**
```
T_{α→β}(E) = Tr[Γ_α(E) G(E) Γ_β(E) G†(E)]
```

where:
- `G(E) = (E - H + iη)^(-1)` is the retarded Green's function
- `Γ_α(E) = i[Σ_α(E) - Σ_α†(E)]` is the coupling to lead α

**Two-Probe Transmission:**
```
T(E) = Tr[Γ_L(E) G(E) Γ_R(E) G†(E)]
```

**RGF Recursive Equations:**

For a system partitioned into slices:
```
┌─────┬─────┬─────┬─────┬─────┐
│  L  │  1  │  2  │ ... │  R  │
└─────┴─────┴─────┴─────┴─────┘
```

**Forward sweep (left to right):**
```
g₁ = (E - H₁ - Σ_L)^(-1)

g_i = (E - H_i - V_{i,i-1} g_{i-1} V_{i-1,i})^(-1),  i > 1
```

**Backward sweep (right to left):**
```
G_N = g_N

G_i = g_i + g_i V_{i,i+1} G_{i+1} V_{i+1,i} g_i,  i < N
```

**Conductance:**
```
G_cond = (e²/h) T(E_F)
```

The transport functionality (part of Phase 4) provides both a class-based interface for general use and a specialized function for heterostructure stacking:

```python
# transport/rgf.py

class RGF:
    """Recursive Green's Function solver for transport calculations.

    Handles both finite systems (ribbons, disordered systems) and
    layered heterostructures with arbitrary stacking.
    """

    def __init__(self, H: BaseTensor):
        """Initialize RGF with Hamiltonian.

        Parameters:
        -----------
        H : BaseTensor
            Real-space Hamiltonian with labels=['site_i', 'site_j']
            or ['layer', 'site_i', 'site_j'] for layered systems
        """
        self.H = H

    def compute_greens_function(self, energy: float, eta: float = 1e-6) -> BaseTensor:
        """Compute Green's function G(E) = (E - H + i*eta)^(-1).

        Returns:
        --------
        G : BaseTensor with same labels as input H
        """
        ...

def stacking_rgf(layers: list[BaseTensor],
                 inter_hoppings: dict[tuple[str, str], BaseTensor],
                 stack_order: list[str],
                 leads: tuple[str, str] | None = None) -> BaseTensor:
    """
    Specialized RGF function for stacked heterostructures.

    This is a convenience wrapper around RGF class that handles
    layer assembly automatically.

    Parameters:
    -----------
    layers: List of BaseTensor for each layer (H_Au, H_M, H_G, ...)
    inter_hoppings: Dict of inter-layer hopping terms
        {('Au', 'M'): H_AM, ('Au', 'G'): H_AG, ('M', 'G'): H_MG, ...}
    stack_order: List specifying layer order
        ['Au', 'Au', 'M', 'G', 'G', 'M', 'Au']
    leads: Tuple of (left_lead, right_lead) layer names for transport

    Returns:
    --------
    G: BaseTensor with full system Green's function
        labels=['layer', 'site_i', 'site_j']

    Example:
    --------
    >>> G = stacking_rgf(layers, inter_hoppings, stack_order, leads=('Au', 'Au'))
    >>> # Then use Transport class to compute transmission
    """
    # Build total Hamiltonian from layers and inter-layer hoppings
    # Apply RGF algorithm for transport
    ...
```

**Example: Au-MoSe2-Graphene heterostructure**:
```python
from condmatTensor.transport import RGF, Transport, stacking_rgf

# Define layers
H_Au = build_fcc_layer(n_sites=100, hopping=-1.0)      # Au layer
H_M = build_mose2_layer(n_sites=50, hopping=-2.0)      # MoSe2 layer
H_G = build_graphene_layer(n_sites=80, hopping=-2.7)    # Graphene layer

# Define inter-layer hoppings
H_AM = build_inter_layer_hopping('Au', 'MoSe2', t=-0.5)
H_AG = build_inter_layer_hopping('Au', 'Graphene', t=-0.3)
H_MG = build_inter_layer_hopping('MoSe2', 'Graphene', t=-0.4)

# Stack order: Au-Au-M-G-G-M-A
layers = [H_Au, H_Au, H_M, H_G, H_G, H_M, H_Au]
inter_hoppings = {
    ('Au', 'M'): H_AM,
    ('Au', 'G'): H_AG,
    ('M', 'G'): H_MG
}
stack_order = ['Au', 'Au', 'M', 'G', 'G', 'M', 'Au']

# Option 1: Use stacking_rgf function (convenience wrapper)
G = stacking_rgf(layers, inter_hoppings, stack_order, leads=('Au', 'Au'))

# Option 2: Use RGF class directly (for more control)
# rgf = RGF(H_total)  # H_total built from layers manually
# G = rgf.compute_greens_function(energy=0.0)

# Transmission
transport = Transport()
transport.compute(G, energy=0.0)  # Store transmission internally
transport.plot()
```

**Example: Kagome lattice transport**:
```python
# In examples/phase4_kagome_transport.py

from condmatTensor.transport import RGF, Transport
from condmatTensor.lattice import BravaisLattice
from condmatTensor.core import BaseTensor
import torch

# Build Kagome lattice (see Phase 1 for complete lattice building example)
lattice, Hr = build_kagome_lattice()

# Build Kagome ribbon (finite system)
H_kagome = build_finite_system(Hr, nx=50, ny=10)

# Add disorder
H_kagome = add_disorder(H_kagome, strength=0.2)

# RGF using class interface
rgf = RGF(H_kagome)
G = rgf.compute_greens_function(energy=0.0)

# Transmission - compute and plot using same instance
transport = Transport()
transport.compute(G, energy=0.0)  # Store transmission internally
transport.plot()  # Uses stored transmission
```

### Phase 6: ML/Bayesian Optimization

**Goal**: Automated parameter search.

**Key Components**:
- `optimization/bayesian.py` for Bayesian optimization
- `optimization/ml_interface.py` for ML surrogate models

**Example**: Optimize U, beta for Kagome Hubbard model (see `examples/phase5_kagome_optimize.py`)
```python
from condmatTensor.optimization.bayesian import BayesianOptimizer
from condmatTensor.manybody import DMFTLoop
from condmatTensor.manybody.preprocessing import generate_matsubara_frequencies
from condmatTensor.manybody import IPTSolver
from condmatTensor.lattice import BravaisLattice, generate_kmesh
from condmatTensor.core import BaseTensor
import torch

# Build Kagome Hubbard model
lattice, Hr = build_centered_kagome_lattice()  # Using generic building blocks
Hk = Hr.to_k_space(k_mesh)
omega = generate_matsubara_frequencies(beta=10.0, n_freq=256)

def objective(params):
    U, beta = params
    solver = IPTSolver(U=U, beta=beta)
    dmft = SingleSiteDMFTLoop(Hk, omega, solver, mu=0.0)
    Sigma = dmft.run()
    # Minimize spectral weight at Fermi level (Mott transition indicator)
    return -Sigma.imag.max()

# Search for Mott transition
opt = BayesianOptimizer(objective, bounds=[(0, 10), (1, 50)])
result = opt.optimize(n_iter=50)
print(f"Optimal U={result.x[0]:.2f}, beta={result.x[1]:.2f}")
```

### Parallel: CNN-Self Attention Selected CI for ED

**Goal**: Improve ED solver efficiency.

**Key Components**:
- `solvers/ed.py` for base ED
- `solvers/ed_cnn.py` for CNN-self attention CI selection

**Architecture**:
```
Input: Hamiltonian matrix H (BaseTensor)
       ↓
CNN (spatial patterns) → Self-Attention (long-range correlations)
       ↓
CI state importance scores → Select top N states
       ↓
ED diagonalization on selected subspace
```

**ED Input Format for Hubbard Models**:

For Hubbard models with interactions, the Hamiltonian representation depends on the approach:

1. **Second Quantized Approach** (recommended for ED):
   - `H_cluster` is a BaseTensor with labels=['site_i', 'site_j']
   - Contains only hopping terms (one-body)
   - Interaction U is passed separately to ED solver
   - ED solver constructs full many-body Hamiltonian internally

2. **Effective Hamiltonian Approach**:
   - `H_cluster` includes interaction effects as mean-field terms
   - U is incorporated into on-site potentials
   - BaseTensor with labels=['site_i', 'site_j'] is sufficient

```python
# Recommended approach for Hubbard models
H_hopping = build_kagome_cluster(lattice, nx=2, ny=2, t=-1.0)
# H_hopping: BaseTensor with labels=['site_i', 'site_j']

# ED solver handles U separately
ed = ED(H_hopping, U=4.0, n_electrons=12)  # U passed as parameter
```

**Example**: Compare ED vs CNN-CI on Kagome cluster (see `examples/parallel_ed_cnn.py`)
```python
from condmatTensor.manybody import ED, CNN_CI_ED
from condmatTensor.lattice import BravaisLattice
from condmatTensor.core import BaseTensor
import torch

# Build Kagome cluster (12 sites)
# Build lattice (see Phase 1 for complete Kagome lattice example)
lattice, _ = build_kagome_lattice()

# Build finite cluster Hamiltonian (hopping only)
H_cluster = build_kagome_cluster(lattice, nx=2, ny=2, t=-1.0)
# H_cluster is BaseTensor with labels=['site_i', 'site_j']
# Note: U is passed to ED solver, not included in H_cluster

# Full ED (exact)
ed_full = ED(H_cluster, U=4.0, n_electrons=12)
E_full, psi_full = ed_full.solve()

# CNN-CI ED (approximate, faster)
ed_cnn = CNN_CI_ED(H_cluster, U=4.0, n_electrons=12, n_selected=1000)
E_cnn, psi_cnn = ed_cnn.solve()

# Compare
print(f"Full ED ground state energy: {E_full[0]:.6f}")
print(f"CNN-CI ground state energy: {E_cnn[0]:.6f}")
print(f"Error: {abs(E_cnn[0] - E_full[0]):.6e}")
```

### Phase 10: Symmetry Reduction for Memory Efficiency

**Corresponding Example**: `examples/phase10_symmetry_reduction.py`

**Goal**: Exploit lattice symmetries to reduce k-mesh size and memory usage.

**Key Components**:
- `lattice/symmetry.py` with `PointGroup`, `SymmetryReducer` classes
- `generate_ibz_kmesh()` for irreducible Brillouin zone generation
- Symmetry-weighted summation for observables

**When to Use**: After LEVEL 8 (interface) is complete, for large-scale calculations where memory is limiting.

**Formalism**:
```
Full BZ sum:     (1/N_k) Σ_{k∈BZ} f(k)
IBZ sum:         Σ_{k∈IBZ} w_k · f(k)
Memory savings:  N_BZ / N_IBZ
```

**Example**: Square lattice with C₄v symmetry → 8x k-mesh reduction
```python
from condmatTensor.lattice import BravaisLattice
from condmatTensor.lattice.symmetry import PointGroup, SymmetryReducer, generate_ibz_kmesh
from condmatTensor.core import BaseTensor
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import DOSCalculator
import torch

# Build square lattice
a = 1.0
cell_vectors = torch.tensor([[a, 0.0], [0.0, a]])
basis_positions = [torch.tensor([0.0, 0.0])]
lattice = BravaisLattice(cell_vectors, basis_positions, num_orbitals=1)

# Option 1: Automatic IBZ generation (recommended)
# -------------------------------------------------
# Detect point group and generate IBZ automatically
reducer = SymmetryReducer(lattice)

# Generate IBZ k-mesh with symmetry weights
k_ibz, weights = generate_ibz_kmesh(lattice, nk=64)
# k_ibz: (N_ibz, 2), weights: (N_ibz,)
# For square lattice: 64² = 4096 → N_ibz ≈ 512 (8x reduction)

# Compute H(k) on IBZ only
Hk_ibz = Hr.to_k_space(k_ibz)  # BaseTensor with labels=['k', 'orb_i', 'orb_j']

# Diagonalize on IBZ
E_ibz = diagonalize(Hk_ibz)[0]  # Shape: (N_ibz, N_band)

# Apply symmetry weights for DOS
dos = DOSCalculator()
omega = torch.linspace(-4, 4, 500)
# Symmetry-weighted DOS: ρ(ω) = Σ_k w_k · (η/π) / [(ω - ε)² + η²]
dos.from_eigenvalues(E_ibz, omega, eta=0.05, weights=weights)
dos.plot()

# Option 2: Explicit point group specification
# -------------------------------------------
# For lattices where automatic detection fails
pg = PointGroup(group_name='C4v')  # Square lattice symmetry
k_ibz, weights = pg.generate_ibz(lattice, nk=64)

# Option 3: Time-reversal symmetry only
# --------------------------------------
# For systems without spatial symmetry
pg = PointGroup(time_reversal=True)  # Only TRS
k_ibz, weights = pg.generate_ibz(lattice, nk=64)
# For TRS only: 2x reduction (k and -k are equivalent)

# Memory comparison
print(f"Full BZ k-points: {64**2}")
print(f"IBZ k-points: {len(k_ibz)}")
print(f"Memory reduction: {64**2 / len(k_ibz):.1f}x")
```

**3D Lattice Example (using spglib)**:
```python
from condmatTensor.lattice import BravaisLattice
from condmatTensor.lattice.symmetry import SymmetryReducer3D, generate_ibz_kmesh_3d
from condmatTensor.core import BaseTensor
from condmatTensor.solvers import diagonalize
import torch

# Build 3D FCC lattice
a = 1.0
cell_vectors = torch.tensor([
    [0.0, a/2, a/2],
    [a/2, 0.0, a/2],
    [a/2, a/2, 0.0]
])
basis_positions = [
    torch.tensor([0.0, 0.0, 0.0]),
    torch.tensor([0.25, 0.25, 0.25])  # FCC basis
]
lattice = BravaisLattice(cell_vectors, basis_positions, num_orbitals=1)

# Atomic numbers for spglib
numbers = [1] * len(basis_positions)  # All same element

# SymmetryReducer3D uses spglib internally
reducer = SymmetryReducer3D(lattice, numbers)

# Get space group info (via spglib)
spacegroup = reducer.get_spacegroup()
print(f"Space group: {spacegroup['international']}")  # e.g., "Fm-3m"

# Generate IBZ k-mesh using spglib symmetry operations
k_ibz, weights = generate_ibz_kmesh_3d(lattice, nk=32, numbers=numbers)
# k_ibz: (N_ibz, 3), weights: (N_ibz,)
# For FCC: 32³ = 32768 → N_ibz ≈ 4096 (8x reduction)

# Compute on IBZ
Hk_ibz = Hr.to_k_space(k_ibz)
E_ibz = diagonalize(Hk_ibz)[0]

# Apply symmetry weights
dos = DOSCalculator()
dos.from_eigenvalues(E_ibz, omega, eta=0.05, weights=weights)
dos.plot()

print(f"Full BZ k-points: {32**3}")
print(f"IBZ k-points: {len(k_ibz)}")
print(f"Memory reduction: {32**3 / len(k_ibz):.1f}x")
```

**2D vs 3D Symmetry Detection**:
| Dimension | Method | Library | Notes |
|-----------|--------|---------|-------|
| 2D | Analytic | Self-implemented | Detects point group from cell_vectors |
| 3D | Space group | spglib | Robust detection of 230 space groups |

**Supported Point Groups (2D, self-implemented)**:
- **Cyclic**: C₂, C₃, C₄, C₆ (n-fold rotation)
- **Dihedral**: D₂, D₃, D₄, D₆ (rotation + reflections)
- **With TRS**: Add time-reversal symmetry for additional reduction

**Supported Space Groups (3D, via spglib)**:
- All 230 space groups supported
- Automatic detection from atomic positions
- Includes symmorphic and non-symmorphic groups

**Memory Savings Examples**:
| Lattice | Dimension | Symmetry | Reduction |
|---------|-----------|----------|-----------|
| Square | 2D | D₄h + TRS | ~8x |
| Hexagonal | 2D | D₆h + TRS | ~12x |
| Triangular | 2D | D₃h + TRS | ~6x |
| Rectangular | 2D | C₂v + TRS | ~4x |
| FCC | 3D | Oh + TRS | ~8x |
| BCC | 3D | Oh + TRS | ~8x |
| Simple Cubic | 3D | Oh + TRS | ~8x |

**Symmetry-Weighted Integration**:
```python
# For any BZ integral, replace sum with weighted sum
def bz_integral_symmetric(f_k, weights):
    """Compute ∫_BZ f(k) dk using symmetry weights."""
    return torch.sum(weights[:, None] * f_k, dim=0)

# Example: Chern number with symmetry
def chern_number_symmetric(Q, k_ibz, weights, n_bands):
    """Compute Chern number using IBZ."""
    Omega = 2 * Q.imag  # Berry curvature from QGT
    # Symmetry-weighted integral over IBZ
    C = torch.sum(weights * Omega[:, :n_bands]) / (2 * torch.pi)
    return C.item()
```

**Implementation Notes**:
- LEVEL 10 is implemented AFTER LEVEL 8 (interface) is complete
- **2D lattices**: Self-implemented analytic point group detection (no external dependency)
- **3D lattices**: Requires spglib for space group detection (robust, supports all 230 groups)
- Requires full lattice information (cell_vectors, basis_positions)
- For 3D: atomic numbers required for spglib
- Weights are normalized: Σ_k w_k = 1
- Works with all analysis modules (DOS, Chern, AHE, etc.)

**API Design**:
```python
# 2D: Self-implemented, no external dependency
from condmatTensor.lattice.symmetry import SymmetryReducer2D
reducer_2d = SymmetryReducer2D(lattice)  # Automatic point group detection

# 3D: Uses spglib
from condmatTensor.lattice.symmetry import SymmetryReducer3D
reducer_3d = SymmetryReducer3D(lattice, atomic_numbers)  # spglib detection

# Or use unified interface (auto-detects dimensionality)
from condmatTensor.lattice.symmetry import SymmetryReducer
reducer = SymmetryReducer(lattice, atomic_numbers=numbers)
```

## Tensor Shapes Reference

| Object | Shape | Labels | Example |
|--------|-------|--------|---------|
| cell_vectors | (dim, dim) | ['a', 'dim'] | (2, 2) for 2D |
| displacements | (N_R, dim) | ['R', 'dim'] | (5, 2) for 5 hoppings |
| H(R) | (N_R, N_orb, N_orb) | ['R', 'orb_i', 'orb_j'] | (5, 2, 2) |
| H(k) | (N_k, N_orb, N_orb) | ['k', 'orb_i', 'orb_j'] | (1024, 2, 2) |
| k_ibz | (N_ibz, dim) | ['k', 'dim'] with weights attribute | (512, 2) for 64² BZ |
| G(k,iωₙ) | (N_k, N_orb, N_orb, N_ω) | ['k', 'orb_i', 'orb_j', 'iwn'] | (1024, 2, 2, 256) |
| Σ(iωₙ) | (N_ω,) | ['iwn'] (pure imaginary) | (256,) |
| Σ_cl(iωₙ) | (N_c, N_c, N_ω) | ['c_i', 'c_j', 'iwn'] | (4, 4, 256) |
| G_layer | (N_k, N_band, N_band, dim) | ['k', 'band_i', 'band_j', 'dim'] | (1024, 3, 3, 2) |
| QGT | (N_k, N_band, N_band, dim, dim) | ['k', 'band_i', 'band_j', 'dim_i', 'dim_j'] | (1024, 3, 3, 2, 2) |
| Berry curvature Ω | (N_k, N_band) | ['k', 'band'] | (1024, 3) |
| symmetry_weights | (N_ibz,) | [] (standalone tensor) | (512,) normalized: Σw_k = 1 |

**QGT Processing for Chern Number**:
- `chern_number(Q, k_mesh, n_bands)` extracts the antisymmetric (imaginary) part of QGT
- Integrates over Brillouin zone: C = (1/2π) ∫_BZ Ω_n(k) d²k
- For 2D systems: Ω_n(k) = 2 * Im[Q_nn^(xy)(k)]

## Dependencies

```
torch>=2.0
numpy>=1.24
matplotlib>=3.7
plotly>=5.0              # For interactive plotting
scipy>=1.10
pyyaml>=6.0
scikit-optimize>=0.9    # For Bayesian optimization
spglib>=2.0             # For 3D space group detection (LEVEL 10)
```

**Note on spglib**: Required only for LEVEL 10 (symmetry) with 3D lattices. 2D symmetry detection uses self-implemented analytic methods.

**Plotting Backends:**
- **plotly** (preferred): Interactive plots, 3D visualization of Fermi surfaces, Berry curvature
- **matplotlib** (fallback): Static plots, publication-quality figures
- Auto-detection: Falls back to matplotlib if plotly unavailable

## Design Principles

1. **Tensor-First**: All physics data in `BaseTensor`
2. **Unified Interface**: H, G, Σ all use same class
3. **GPU-Ready**: Automatic chunking and device management
4. **Plugin Architecture**: Extensible for different input methods
5. **Integrated Plotting**: Each analysis module has its own `plot()`
6. **Clear Dependencies**: Explicit `__init__.py` exports
7. **Layered Architecture**: No circular imports, progressive complexity

## Development Rules

### Rule 1: Virtual Environment Setup

All development must be done under a dedicated virtual environment:

```bash
# Create virtual environment
python -m venv env_condmatTensor

# Activate (Linux/Mac)
source env_condmatTensor/bin/activate

# Activate (Windows)
env_condmatTensor\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Rationale**: Ensures consistent dependency isolation and reproducible development environment.

### Rule 2: Test-After-Module with Kagome Example

After completing each module (LEVEL), run the Kagome example to verify correctness(>> examples/kagome_ex_results. output them into the folder):

```python
# After finishing a module, test with:
from condmatTensor.lattice import BravaisLattice, generate_kmesh, generate_k_path
from condmatTensor.core import BaseTensor
from condmatTensor.analysis import DOSCalculator, BandStructure
import torch

# Build Kagome lattice
lattice, Hr = build_kagome_lattice()

# 1. Test along k-path (plot analytic vs computed)
k_path = generate_k_path(lattice, ['G', 'K', 'M', 'G'], 100)
Hk_path = Hr.to_k_space(k_path)
E_path, U_path = diagonalize(Hk_path)

# 2. Test on k-mesh (compute observables)
k_mesh = generate_kmesh(lattice, 50)
Hk_mesh = Hr.to_k_space(k_mesh)
E_mesh = diagonalize(Hk_mesh)[0]

# Plot results for verification
bs = BandStructure()
bs.compute(k_path, E_path)
bs.plot(labels=['G', 'K', 'M', 'G'])

dos = DOSCalculator()
dos.from_eigenvalues(E_mesh, torch.linspace(-4, 4, 500), eta=0.05)
dos.plot()

# Verify: Analytic Kagome flat band at ε = -2t
# Verify: DOS divergence at Van Hove singularities
```
Refer to existed kagome examples, there are four important models.

**Required Plots After Each Module:**
- Band structure along high-symmetry k-path
- DOS with analytic comparison where available
- Any new observable added by the module

### Rule 3: Minimize CPU-GPU Data Transfer

Control the cost of data transfer between CPU and GPU:

```python
# GOOD: Minimize transfers
device = get_device()  # Auto-detect CUDA, fall back to CPU
Hk = Hk.to(device)     # Transfer once
result = computation_on_gpu(Hk)  # All computation on GPU
result_cpu = result.cpu()  # Transfer back once at end

# BAD: Frequent transfers
Hk = Hk.cuda()
for i in range(100):
    Hk = Hk.cpu()  # Unnecessary transfer each iteration
    processed = process(Hk)
    Hk = processed.cuda()
```

**Guidelines:**
- Use `with torch.no_grad():` for inference (no gradient tracking)
- Use `torch.cuda.amp.autocast()` for mixed precision when appropriate
- Chunk large k-meshes to avoid OOM: `choose_chunk_size(total_size, available_memory)`
- Keep tensors on GPU until final result is needed
- Use `pin_memory=True` for data loaders

**GPU Utils API:**
```python
# core/gpu_utils.py

def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > MPS > CPU)."""

def choose_chunk_size(n_total: int, memory_gb: float = None) -> int:
    """Calculate optimal chunk size to avoid OOM."""

def estimate_tensor_memory(shape: tuple, dtype: torch.dtype) -> float:
    """Estimate memory in GB for a tensor."""
```

### Rule 4: Many-Body Algorithm Reference Discussion

For any many-body algorithm implementation, discuss and verify against literature BEFORE coding:

**Required Pre-Coding Checklist:**
1. [ ] Find primary reference paper (preferably < 5 years old for verification)
2. [ ] Identify benchmark test cases (where available)
3. [ ] Extract formalism and equations (add to `FORMALISM.md` if new)
4. [ ] Discuss implementation approach with team/reference
5. [ ] Create test case with known analytic result
6. [ ] Implement and verify against benchmark
7. [ ] Document in code comments with citation

**Algorithms Requiring Discussion:**
- DMFT self-consistency loop
- IPT (Iterated Perturbation Theory) solver
- QGT computation (both autograd and analytic methods)
- Chern number / Z₂ invariant calculation
- RGF (Recursive Green's Function) transport
- ED (Exact Diagonalization) with CNN-selected CI

**Example Documentation Format:**
```python
def ipt_self_energy(G: BaseTensor, U: float, beta: float) -> BaseTensor:
    """Compute IPT self-energy Σ(iωₙ).

    Algorithm: Second-order iterated perturbation theory
    Reference: Merino & Parcollet, Phys. Rev. B 104, 035160 (2021)
    Equation: Σ(iω) = U² · χ₀(iω) · G(iω)

    Implementation verified against:
    - Single-impurity Anderson model (SIAM) analytic solution
    - DMFT convergence for Hubbard model at half-filling

    Args:
        G: Local Green's function, labels=['iwn']
        U: Hubbard interaction strength
        beta: Inverse temperature

    Returns:
        Σ: Self-energy, labels=['iwn'] (pure imaginary)
    """
```

**Verification Protocol:**
```python
# Before merging any many-body algorithm:
def test_ipt_benchmark():
    """Verify IPT against SIAM analytic solution."""
    # Test case: U=2, β=10, known analytic G(iω)
    G_computed = ipt_solver(...)
    G_analytic = load_siam_benchmark("U2_beta10.dat")
    assert torch.allclose(G_computed, G_analytic, atol=1e-3)
```

### Rule 5: Development Logging

All development work must be logged in the `developLog/` directory to maintain a comprehensive record of project evolution.

**Required Logging Structure:**

```bash
developLog/
├── allAPI.md              # Complete API documentation (all modules, examples, parameters)
├── developLog_YYYY-MM-DD.md  # Daily development log entries
└── ...
```

**Required Contents:**

1. **`developLog/allAPI.md`** - Complete API documentation containing:
   - All modules organized by level (1-10)
   - Usage examples for each method/class
   - API descriptions with full parameters
   - Return types and behavior
   - Cross-references between related modules

2. **`developLog/developLog_YYYY-MM-DD.md`** - Daily development logs containing:
   - Date of work session
   - Summary of changes made
   - New features implemented
   - Bugs fixed or issues resolved
   - API changes or deprecations
   - Validation/test results
   - Next steps or pending items

**When to Create/Update Logs:**

- Create a new daily log file when starting development work on a new day
- Update `allAPI.md` whenever:
  - A new module is implemented
  - A new class/function is added
  - API signatures change
  - New examples are added

**Logging Template:**

```markdown
# Development Log - YYYY-MM-DD

## Summary
[Brief description of work done]

## Changes Made
- [ ] Feature 1
- [ ] Feature 2

## API Changes
- New: `function_name()` - Description
- Changed: `ClassName.method()` - Modified behavior
- Deprecated: `old_function()` - Use `new_function()` instead

## Validation
- [ ] Test passed: description
- [ ] Benchmark: results

## Next Steps
1. Item 1
2. Item 2
```

**Rationale**: Maintains comprehensive project history, aids in onboarding new developers, and provides traceability for all changes.

**Cross-Reference Files:**
- This rule is also documented in: `CLAUDE.md`, `.cursorrules`

## Architecture Comparison with Reference Libraries

### Comparison Table

| Aspect | NumPy | WannierTools | TRIQS | condmatTensor |
|--------|-------|--------------|-------|---------------|
| **Core Language** | C/Fortran | Fortran | C++ | Python |
| **Python API** | Native C-API | Optional wrapper | pybind11/Boost | Native |
| **Primary Backend** | BLAS/LAPACK | BLAS/LAPACK | HDF5/MPI | PyTorch |
| **Dependency Style** | Layered (core→submodules) | Monolithic Fortran | Building blocks | Layered (9 levels) |
| **GPU Support** | Via CuPy/compat | No | No (CPU) | Yes (PyTorch native) |
| **Input Method** | Programmatic | File-driven (wt.in) | Programmatic | Programmatic |
| **Domain** | General arrays | Tight-binding topology | Many-body DMFT | Condensed matter |

### Design Pattern Adoption

| Pattern | Source | Applied in condmatTensor |
|---------|--------|-------------------------|
| **Layered Architecture** | NumPy | 9-level dependency hierarchy |
| **Building Blocks** | TRIQS | Composable `BaseTensor` + lattice classes |
| **Explicit Exports** | NumPy | `__all__` in each `__init__.py` |
| **Type Annotations** | TRIQS C++ | Full Python type hints |
| **Progressive Disclosure** | NumPy | Core in main namespace, advanced in submodules |

## API Reference: Complete Type Hints

This section provides complete type specifications for all critical functions and classes.

### Core Module

```python
from typing import Optional, List
import torch

class BaseTensor:
    def __init__(
        self,
        tensor: torch.Tensor,
        labels: List[str],
        orbital_names: Optional[List[str]] = None,
        displacements: Optional[torch.Tensor] = None
    ) -> None:
        """Initialize BaseTensor.

        Args:
            tensor: Underlying tensor data
            labels: Semantic labels for each dimension
            orbital_names: Physical names of orbitals (e.g., ['px', 'py'])
            displacements: Real-space displacements for H(R) tensors

        Attributes:
            tensor: torch.Tensor - the data
            labels: List[str] - dimension labels
            orbital_names: Optional[List[str]] - orbital names
            displacements: Optional[torch.Tensor] - shape (N_R, dim)
        """

    def to_k_space(self, k: torch.Tensor) -> 'BaseTensor':
        """Fourier transform from R-space to k-space.

        Args:
            k: K-points, shape (N_k, dim)

        Returns:
            BaseTensor with labels=['k', 'orb_i', 'orb_j'] (or similar)

        Raises:
            ValueError: If displacements is None or labels don't contain 'R'
        """
```

### Lattice Module

```python
class BravaisLattice:
    def __init__(
        self,
        cell_vectors: torch.Tensor,  # (dim, dim)
        basis_positions: List[torch.Tensor],
        num_orbitals: int
    ) -> None:
        """Bravais lattice with multiple sites per unit cell.

        Args:
            cell_vectors: Lattice vectors, shape (dim, dim)
            basis_positions: List of basis atom positions, each (dim,)
            num_orbitals: Number of orbitals per site
        """

def generate_kmesh(lattice: BravaisLattice, nk: int) -> torch.Tensor:
    """Generate uniform k-mesh.

    Returns:
        torch.Tensor of shape (nk^dim, dim)
    """

def generate_k_path(
    lattice: BravaisLattice,
    points: List[str],
    n_per_segment: int
) -> torch.Tensor:
    """Generate k-point path along high-symmetry lines.

    Args:
        points: List of high-symmetry point labels (e.g., ['G', 'K', 'M'])
        n_per_segment: Points per segment

    Returns:
        torch.Tensor of shape (n_total, dim)
    """
```

### DMFT Preprocessing Module

```python
def generate_kmesh(lattice: BravaisLattice, nk: int) -> torch.Tensor:
    """Generate uniform k-mesh for DMFT.

    Returns:
        torch.Tensor of shape (nk^dim, dim)
    """

def generate_matsubara_frequencies(beta: float, n_freq: int) -> torch.Tensor:
    """Generate fermionic Matsubara frequencies.

    Args:
        beta: Inverse temperature
        n_freq: Number of positive frequencies

    Returns:
        torch.Tensor of shape (2*n_freq + 1,) - includes negative and zero
        labels: ['omega_n']
    """

def initialize_self_energy(n_freq: int, n_orb: int) -> BaseTensor:
    """Initialize self-energy tensor.

    Returns:
        BaseTensor with shape (n_freq,), labels=['omega']
    """
```

### DMFT Single-Site Module

```python
class SingleSiteDMFTLoop:
    def __init__(
        self,
        Hk: BaseTensor,  # labels=['k', 'orb_i', 'orb_j']
        omega: torch.Tensor,  # Matsubara frequencies
        solver: IPTSolver,
        mu: float = 0.0
    ) -> None:
        """Initialize DMFT loop.

        Args:
            Hk: k-space Hamiltonian
            omega: Matsubara frequency grid
            solver: Impurity solver instance
            mu: Chemical potential
        """

    def run(
        self,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> BaseTensor:
        """Run DMFT self-consistency loop.

        Returns:
            BaseTensor with shape (n_omega,), labels=['omega']
            Contains converged self-energy Σ(iω)
        """
```

### QGT Module

```python
def hk_to_g_layer(Hk: BaseTensor, k: torch.Tensor) -> BaseTensor:
    """Convert H(k) to position operator in band basis.

    Args:
        Hk: BaseTensor with labels=['k', 'orb_i', 'orb_j']
             orbital_names may be set (e.g., ['px', 'py', 'pz'])
        k: K-point mesh, shape (N_k, dim)

    Returns:
        BaseTensor with labels=['k', 'band_i', 'band_j', 'dim']
        orbital_names=None (lost during orbital→band transformation)
        Shape: (N_k, N_band, N_band, dim)
    """

def hr_to_g_layer(
    hr_tensor: BaseTensor,
    displacements: torch.Tensor,
    k: torch.Tensor
) -> BaseTensor:
    """Convert H(R) to position operator using analytic derivatives.

    Args:
        hr_tensor: BaseTensor with labels=['R', 'orb_i', 'orb_j']
                    Must have displacements attribute set
        displacements: Displacement vectors, shape (N_R, dim)
        k: K-point mesh, shape (N_k, dim)

    Returns:
        BaseTensor with labels=['k', 'band_i', 'band_j', 'dim']
        orbital_names=None (lost during transformation)
    """

def compute_qgt(g_layer: BaseTensor) -> BaseTensor:
    """Compute Quantum Geometric Tensor from G_layer.

    Args:
        g_layer: BaseTensor with labels=['k', 'band_i', 'band_j', 'dim']

    Returns:
        BaseTensor with labels=['k', 'band_i', 'band_j', 'dim_i', 'dim_j']
        Shape: (N_k, N_band, N_band, dim, dim)
        Contains both Berry curvature (imaginary) and metric (real)
    """
```

### Topology Module

```python
def berry_curvature(Q: BaseTensor) -> BaseTensor:
    """Extract Berry curvature from QGT.

    Args:
        Q: BaseTensor with labels=['k', 'band_i', 'band_j', 'dim_i', 'dim_j']

    Returns:
        BaseTensor with labels=['k', 'band']
        Shape: (N_k, N_band)
        Contains Ω_n(k) for each band
    """

def chern_number(
    Q: BaseTensor,
    k_mesh: torch.Tensor,
    n_bands: int
) -> float:
    """Compute Chern number from QGT.

    Args:
        Q: BaseTensor with labels=['k', 'band_i', 'band_j', 'dim_i', 'dim_j']
        k_mesh: K-point mesh, shape (N_k, dim)
        n_bands: Number of bands to consider

    Returns:
        float: Total Chern number (sum over occupied bands)

    Note:
        Integrates Ω_n(k) = 2 * Im[Q_nn^(xy)(k)] over BZ
    """

def ahe(
    Q: BaseTensor,
    E: torch.Tensor,  # Eigenvalues, shape (N_k, N_band)
    mu: float,
    k_mesh: torch.Tensor
) -> float:
    """Compute anomalous Hall conductivity.

    Args:
        Q: BaseTensor with labels=['k', 'band_i', 'band_j', 'dim_i', 'dim_j']
        E: Eigenvalues at each k-point, shape (N_k, N_band)
        mu: Chemical potential (Fermi level)
        k_mesh: K-point mesh, shape (N_k, dim)

    Returns:
        float: σ_xy in units of e²/h
    """
```

### Transport Module

```python
class RGF:
    def __init__(self, H: BaseTensor) -> None:
        """Initialize RGF solver.

        Args:
            H: Real-space Hamiltonian with labels=['site_i', 'site_j']
               or labels=['layer', 'site_i', 'site_j'] for layered systems
        """

    def compute_greens_function(
        self,
        energy: float,
        eta: float = 1e-6
    ) -> BaseTensor:
        """Compute Green's function G(E) = (E - H + i*eta)^(-1).

        Returns:
            BaseTensor with same labels as input H
        """

def stacking_rgf(
    layers: List[BaseTensor],
    inter_hoppings: dict[tuple[str, str], BaseTensor],
    stack_order: List[str],
    leads: Optional[tuple[str, str]] = None
) -> BaseTensor:
    """RGF for stacked heterostructures.

    Args:
        layers: List of layer Hamiltonians, each with labels=['site_i', 'site_j']
        inter_hoppings: Dict mapping (layer1, layer2) -> hopping tensor
        stack_order: List of layer names in stacking order
        leads: (left_lead, right_lead) tuple for transport

    Returns:
        BaseTensor with labels=['layer', 'site_i', 'site_j']
    """

class Transport:
    def __init__(self) -> None:
        """Transport calculator."""

    def compute(
        self,
        G: BaseTensor,
        energy: float
    ) -> torch.Tensor:
        """Compute transmission.

        Args:
            G: Green's function from RGF
            energy: Energy at which to compute transmission

        Returns:
            torch.Tensor of transmission coefficients (stored internally)
        """
```

### Analysis Modules

```python
class DOSCalculator:
    def __init__(self) -> None:
        """DOS calculator with internal storage."""
        self.omega: Optional[torch.Tensor] = None
        self.rho: Optional[torch.Tensor] = None

    def from_eigenvalues(
        self,
        E_k: torch.Tensor,  # shape (N_k, N_band)
        omega: torch.Tensor,
        eta: float = 0.02
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute DOS from eigenvalues.

        Returns:
            (omega, rho) tuple - also stored in self.omega, self.rho
        """

    def plot(self, ax: Optional[Any] = None, **kwargs) -> Any:
        """Plot stored DOS results."""
```

## Import Path Reference

| Component | Import Path |
|-----------|-------------|
| BaseTensor | `from condmatTensor.core import BaseTensor` |
| BravaisLattice | `from condmatTensor.lattice import BravaisLattice` |
| generate_kmesh | `from condmatTensor.lattice import generate_kmesh` |
| DMFTLoop | `from condmatTensor.manybody import DMFTLoop` |
| ClusterDMFTLoop | `from condmatTensor.manybody import ClusterDMFTLoop` |
| generate_matsubara_frequencies | `from condmatTensor.manybody.preprocessing import generate_matsubara_frequencies` |
| IPTSolver | `from condmatTensor.manybody import IPTSolver` |
| ED | `from condmatTensor.manybody import ED` |
| CNN_CI_ED | `from condmatTensor.manybody import CNN_CI_ED` |
| YAMLReader | `from condmatTensor.interface.yaml_reader import YAMLReader` |
| WannierReader | `from condmatTensor.interface.wannier_reader import WannierReader` |
| hk_to_g_layer | `from condmatTensor.analysis.qgt import hk_to_g_layer` |
| compute_qgt | `from condmatTensor.analysis.qgt import compute_qgt` |
| chern_number | `from condmatTensor.analysis.topology import chern_number` |
| RGF | `from condmatTensor.transport import RGF` |
| stacking_rgf | `from condmatTensor.transport import stacking_rgf` |
| BayesianOptimizer | `from condmatTensor.optimization.bayesian import BayesianOptimizer` |
| DOSCalculator | `from condmatTensor.analysis import DOSCalculator` |

## References

[1] G. Kotliar and D. Vollhardt, "Dynamical mean-field theory of strongly correlated fermion systems and the limit of infinite dimensions", *Rev. Mod. Phys.* **76**, 903 (2004).

[2] A. Georges, G. Kotliar, W. Krauth, and M. J. Rozenberg, "Dynamical mean-field theory of strongly correlated fermion systems and the limit of infinite dimensions", *Rev. Mod. Phys.* **68**, 13 (1996).

[3] H. Kajueter, "PhD Thesis: Iterated Perturbation Theory for the Hubbard Model" (1996).

[4] J. Merino and O. Parcollet, "Iterated perturbation theory for the Hubbard model", *Phys. Rev. B* **104**, 035160 (2021).

[5] N. Nagaosa, J. Sinova, S. Onoda, A. H. MacDonald, and N. P. Ong, "Anomalous Hall effect", *Rev. Mod. Phys.* **82**, 1539 (2010).

[6] D. Xiao, M.-C. Chang, and Q. Niu, "Berry phase effects on electronic properties", *Rev. Mod. Phys.* **82**, 1959 (2010).

[7] H. Fukui, T. Hatsugai, and H. Suzuki, "Chern numbers in discretized Brillouin zone", *J. Phys. Soc. Jpn.* **74**, 1674 (2005).

[8] S. Datta, *Electronic Transport in Mesoscopic Systems* (Cambridge, 1995).

[9] Y. V. Nazarov and Y. M. Blanter, *Quantum Transport* (Cambridge, 2009).

[10] G. D. Mahan, *Many-Particle Physics*, 3rd ed. (Kluwer, 2000).

[11] A. L. Fetter and J. D. Walecka, *Quantum Theory of Many-Particle Systems* (Dover, 2003).

[12] A. Altland and B. Simons, *Condensed Matter Field Theory*, 2nd ed. (Cambridge, 2010).

[13] C. M. Goringe, D. R. Bowler, and E. Hernández, "Tight-binding modelling of materials", *Rep. Prog. Phys.* **60**, 1447 (1997).

[14] E. Gull, A. J. Millis, A. I. Lichtenstein, A. N. Rubtsov, M. Troyer, and P. Werner, "Continuous-time quantum Monte Carlo methods for quantum impurity models", *Rev. Mod. Phys.* **83**, 349 (2011).

[15] M. V. M. Deserno, "Mathematical foundations of tight-binding models", *Phys. Rep.* **2024**.