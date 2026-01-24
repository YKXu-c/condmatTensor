# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Comprehensive guide for Claude Code instances working on the condmatTensor repository.**
> For detailed architecture reference, see [`plans/architecture_plan.md`](plans/architecture_plan.md) and [`plans/DEPENDENCY_ANALYSIS.md`](plans/DEPENDENCY_ANALYSIS.md).

---

## Project Overview

**condmatTensor** is a PyTorch-based condensed matter physics library for quantum materials research. The library uses a **unified tensor-first approach** where all physics objects (Hamiltonians, Green's functions, self-energies) are represented by a single `BaseTensor` class with semantic labels.

- **Package Version**: 0.0.1
- **License**: MIT
- **Implementation Status**: ~45% complete (5 of 10 levels partially/fully implemented, ~4,500 lines)
- **Key Innovation**: One `BaseTensor` class for H, G, Σ with automatic R→k Fourier transforms

---

## Critical Installation Order

**PyTorch MUST be installed FIRST via CUDA-specific URL, then remaining dependencies:**

```bash
# Step 1: Install PyTorch 2.10+ with CUDA 13.0 support FIRST
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Step 2: Install remaining dependencies
pip install -r requirements.txt
```

**Why this order?** The official PyTorch version for this project is **PyTorch 2.10+cu13**. Installing from PyPI default may give CPU-only version.

### Dependencies

| Type | Package | Version |
|------|---------|---------|
| **Required** | `torch` | >=2.10 |
| **Required** | `numpy` | >=1.24, <2.0 (numpy<2.0 for sober-bo) |
| **Required** | `matplotlib` | >=3.7 |
| **Required** | `scipy` | >=1.10 |
| **Required** | `pyyaml` | >=6.0 |
| **Optional** (LEVEL 7) | `sober-bo` | ==2.0.4 (preferred) |
| **Optional** (LEVEL 7) | `botorch` | >=0.9.0 (alternative) |
| **Optional** (LEVEL 7) | `scikit-optimize` | >=0.9 (fallback) |

---

## 10-Level Architecture

```
                    torch | numpy | matplotlib | scipy
                              └───────┴──────────────┘
                                        │
              ┌─────────────────────────┴─────────────────┐
              │                                             │
              ▼                                             ▼
    ┌──────────────────┐                        ┌──────────────────┐
    │  LEVEL 1: Core   │                        │  LEVEL 9: Logging│
    │  ✅ Complete     │                        │  ❌ Not started  │
    └─────────┬────────┘                        └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ LEVEL 2: Lattice │
    │ ✅ Complete     │
    └─────────┬────────┘
              │
    ┌─────────┴───────────────────────────────────────────┐
    │                                                     │
    ▼                                                     ▼
┌──────────────┐                                ┌──────────────┐
│LEVEL 3: Solver│                               │LEVEL 8: Interface│
│⚠️ Partial    │                                │  ❌ Not started│
└──────┬───────┘                                └──────┬───────┘
       │                                               │
       ▼                                               ▼
┌───────────────┐                           ┌───────────────┐
│LEVEL 4: Many-Body│                         │LEVEL 10: Symmetry│
│⚠️ Partial     │                           │  ❌ Not started│
└───────┬───────┘                           └───────┬───────┘
        │                                           │
        ▼                   ┌───────────────────────┘
┌───────────────┐           │
│LEVEL 7: Opt   │◄──────────┘
│⚠️ Partial     │
└───────────────┘
    │
    ▼
┌───────────────┐
│LEVEL 6: Transport│
│  ❌ Not started│
└───────────────┘
```

### Implementation Status Summary

| Level | Module | Status | Key Classes |
|-------|--------|--------|-------------|
| **1** | Core | ✅ Complete | `BaseTensor`, `get_device()` |
| **2** | Lattice | ✅ Complete | `BravaisLattice`, `TightBindingModel`, `generate_kmesh`, `generate_k_path` |
| **3** | Solvers | ⚠️ Partial | `diagonalize()` (ED, IPT not implemented) |
| **4** | Many-Body | ⚠️ Partial | `BareGreensFunction`, `SelfEnergy`, `SpectralFunction`, `KondoLatticeSolver` (DMFT loops not implemented) |
| **5** | Analysis | ⚠️ Partial | `DOSCalculator`, `ProjectedDOS`, `BandStructure` (topology, QGT not implemented) |
| **6** | Transport | ❌ Not started | RGF, transport calculations |
| **7** | Optimization | ⚠️ Partial | `BayesianOptimizer`, `EffectiveArrayOptimizer` (ML interface not implemented) |
| **8** | Interface | ❌ Not started | YAML, Wannier90 readers |
| **9** | Logging | ❌ Not started | `CalculationLogger` |
| **10** | Symmetry | ❌ Not started | IBZ reduction |

---

## Common Development Commands

### Running Examples (Validation)
This project uses **example scripts for validation**, NOT pytest.

**Note**: Ensure `PYTHONPATH` is set before running examples:
```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

```bash
# Basic Kagome tight-binding (validates: flat band at E=-2|t|, Dirac points at K)
python examples/kagome_bandstructure.py

# Kagome-F multi-orbital physics
python examples/kagome_with_f_bandstructure.py

# Spinful systems with Zeeman coupling
python examples/kagome_spinful_bandstructure.py

# Bayesian optimization for effective model downfolding
python examples/kagome_f_effective_array.py

# GPU performance benchmark
python examples/gpu_performance_benchmark.py
```

### Virtual Environment
```bash
python -m venv env_condmatTensor
source env_condmatTensor/bin/activate

# Install PyTorch FIRST, then requirements
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# Add package to PYTHONPATH (required for imports)
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

**Note**: The virtual environment `env_condmatTensor` is used during development and is excluded from git.

---

## Key Architectural Concepts

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
# 3 orbital sites × 2 spins = 6 orbitals
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

---

## Import Reference

```python
# LEVEL 1: Core
from condmatTensor.core import BaseTensor, get_device

# LEVEL 2: Lattice
from condmatTensor.lattice import (
    BravaisLattice,
    TightBindingModel,
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
    KondoLatticeSolver,
    SpinFermionModel
)

# LEVEL 5: Analysis
from condmatTensor.analysis import (
    DOSCalculator,
    ProjectedDOS,
    BandStructure
)

# LEVEL 7: Optimization
from condmatTensor.optimization import (
    BayesianOptimizer,
    EffectiveArrayOptimizer
)
```

---

## GPU Device Management

**This is a scientific calculation package. NOT everything runs on GPU.**

### CPU vs GPU Split

| **GPU (Compute-Intensive)** | **CPU (Control Logic & I/O)** |
|---------------------------|------------------------------|
| Bayesian optimization (LEVEL 7) | DMFT loop control (LEVEL 4) |
| Large matrix diagonalization (LEVEL 3) | Plotting with matplotlib |
| Dense tensor operations (all levels) | File I/O (all levels) |
| Lorentzian broadening (LEVEL 5) | Loop overhead (all levels) |
| QGT computations with autograd (LEVEL 5) | Result storage (all levels) |

```python
from condmatTensor.core import get_device

# Auto-detect CUDA with CPU fallback
device = get_device()  # Returns 'cuda' if available, else 'cpu'

# Move tensors/models to GPU
Hk = Hk.to(device)
model = model.to(device)

# Minimize CPU-GPU transfers (transfer once, compute on GPU, transfer back once)

# Pattern: For plotting, always bring back to CPU first
import matplotlib.pyplot as plt
plt.plot(evals.cpu().numpy())  # .cpu() or .numpy() required for matplotlib
```

---

## Bayesian Optimization Backend Priority

`BayesianOptimizer` supports multiple backends with automatic fallback:

| Priority | Backend | Package | Description |
|----------|---------|---------|-------------|
| **1 (Preferred)** | SOBER | `sober-bo==2.0.4` | Sequential Optimization using Ensemble of Regressors |
| **2** | BoTorch | `botorch>=0.9.0` | Gaussian Process with Expected Improvement |
| **3 (Fallback)** | Simple | Python stdlib | Thompson sampling / random search |

```python
from condmatTensor.optimization import BayesianOptimizer

opt = BayesianOptimizer(
    bounds=[(0, 1), (0, 1)],
    backend="auto"  # Tries SOBER > BoTorch > Simple
)
X_best, y_best = opt.optimize(objective_fn, n_init=10, n_iter=50, device=device)
```

---

## Kagome Validation Examples

The `examples/` directory demonstrates and validates core physics:

1. **`kagome_bandstructure.py`** - Pure Kagome lattice
   - Expected: flat band at E=-2|t|, Dirac points at K
   - Validates: basic tight-binding, diagonalization, band structure

2. **`kagome_with_f_bandstructure.py`** - Kagome-F (4 sites/cell)
   - Expected: 8 bands (4×2 spin), f-d hybridization
   - Validates: multi-orbital physics, parameter scans

3. **`kagome_f_effective_array.py`** - Bayesian downfolding
   - Expected: effective J_eff, S_eff parameters
   - Validates: Bayesian optimization, model reduction

---

## Existing Documentation

| File | Description |
|------|-------------|
| [`plans/architecture_plan.md`](plans/architecture_plan.md) | Comprehensive architecture reference (2,604 lines): module details, formalism, equations, workflows, dependencies |
| [`plans/DEPENDENCY_ANALYSIS.md`](plans/DEPENDENCY_ANALYSIS.md) | Comparison with NumPy/TRIQS/WannierTools (847 lines) |
| [`.cursorrules`](.cursorrules) | AI assistant rules for Cursor/Claude Code (751 lines): CPU/GPU split, device selection, development rules |
| [`.github/copilot-instructions.md`](.github/copilot-instructions.md) | GitHub Copilot instructions (365 lines) |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`.gitignore`](.gitignore) | Excludes: `CLAUDE.md`, `plans/`, `env_condmatTensor/`, `.claude/settings.json` |

---

## Development Notes

1. **No pytest configuration** - Validation via example scripts in `examples/`
2. **No build system** - Pure Python, no `setup.py` or `pyproject.toml`
3. **Modular architecture** - Enables progressive implementation
4. **Strong GPU support** - PyTorch native acceleration
5. **Research-focused** - Designed for quantum materials research, not web/applications
6. **No circular dependencies** - Lower levels never depend on higher levels
7. **PYTHONPATH required** - Set environment variable: `export PYTHONPATH=/path/to/condmatTensor/src:$PYTHONPATH`
8. **Development logging** - All development work must be logged in `developLog/` directory:
   - `developLog/allAPI.md` - Complete API documentation (update when API changes)
   - `developLog/developLog_YYYY-MM-DD.md` - Daily development logs
   - See `plans/architecture_plan.md` Rule 5 and `.cursorrules` Rule 8 for details

---

## Quick Reference: Common Workflows

### Band Structure Calculation
```python
from condmatTensor.lattice import BravaisLattice, TightBindingModel, generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure

lattice = BravaisLattice(cell_vectors, basis_positions, num_orbitals)
model = TightBindingModel(lattice, orbital_labels=['A', 'B', 'C'])
model.add_hopping('A', 'B', [0, 0], 1.0)

k_path, labels = generate_k_path(lattice, ['G', 'K', 'M', 'G'], n_k=100)
Hk = model.build_Hk(k_path)
evals, evecs = diagonalize(Hk.tensor)

bs = BandStructure()
bs.compute(evals, k_path, labels)
bs.plot()
```

### Many-Body Green's Function
```python
from condmatTensor.manybody.preprocessing import (
    generate_matsubara_frequencies,
    BareGreensFunction,
    SelfEnergy
)

omega = generate_matsubara_frequencies(beta=10.0, n_max=128)
G0 = BareGreensFunction.from_hamiltonian(Hk, omega, mu=0.0)
Sigma = SelfEnergy(omega, 0.0)
# ... DMFT loop updates Sigma ...
G = G0.apply_self_energy(Sigma)
```

### Bayesian Optimization
```python
from condmatTensor.optimization import BayesianOptimizer

opt = BayesianOptimizer(bounds=[(0, 1), (0, 1)], backend="auto")
X_best, y_best = opt.optimize(objective_fn, n_init=10, n_iter=50, device=device)
```

---

## Git Status Note

The `.gitignore` file excludes `CLAUDE.md`, but this file has been explicitly created per user request to guide future Claude Code instances working in this repository.
