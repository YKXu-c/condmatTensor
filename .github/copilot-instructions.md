# condmatTensor - GitHub Copilot Instructions

> **AI Assistant Guidelines for condmatTensor Development**

This file contains instructions for GitHub Copilot working on the condmatTensor codebase.

## Project Overview

condmatTensor is a PyTorch-based condensed matter physics library for quantum materials research. The library uses a **unified tensor-first approach** where all physics objects (Hamiltonians, Green's functions, self-energies) are represented by a single `BaseTensor` class with semantic labels.

- **Package Version**: 0.0.1
- **Implementation Status**: ~45% complete (5 of 10 levels partially/fully implemented)
- **Key Innovation**: One `BaseTensor` class for H, G, Σ with automatic R→k Fourier transforms

---

## 10-Level Architecture (Mental Model)

```
LEVEL 1: Core      → BaseTensor, get_device() ✅
LEVEL 2: Lattice   → BravaisLattice, HoppingModel ✅
LEVEL 3: Solvers   → diagonalize() (ED, IPT not implemented) ⚠️
LEVEL 4: Many-Body → BareGreensFunction, SelfEnergy (DMFT loops not implemented) ⚠️
LEVEL 5: Analysis  → DOSCalculator, BandStructure (topology, QGT not implemented) ⚠️
LEVEL 6: Transport → ❌ Not started
LEVEL 7: Optimize  → BayesianOptimizer (ML interface not implemented) ⚠️
LEVEL 8: Interface → ❌ Not started
LEVEL 9: Logging   → ❌ Not started
LEVEL 10: Symmetry → ❌ Not started
```

**NO CIRCULAR DEPENDENCIES**: Lower levels never depend on higher levels.

---

## Critical Conventions

### Spinor Convention for Magnetic Systems

**Spin is embedded in orbital indices**: `[orb_0_up, orb_0_down, orb_1_up, orb_1_down, ...]`

```python
# 3 orbital sites × 2 spins = 6 orbitals
num_orbitals = [2, 2, 2]  # Each site has 2 spin states

# Example: Kagome with f-orbital and spin
# [A_up, A_down, B_up, B_down, C_up, C_down, f_up, f_down]
num_orbitals = [2, 2, 2, 2]  # 4 sites × 2 spins = 8 orbitals
```

### BaseTensor - Unified Representation

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

### Fourier Transform Formula

```
H(k) = Σ_R H(R) · exp(i·k·R)
```

- `R`: Real-space displacement vector (Cartesian coordinates)
- `k`: Momentum vector
- Labels change: `['R', 'orb_i', 'orb_j']` → `['k', 'orb_i', 'orb_j']`

### Coordinate Conventions

| Aspect | Convention |
|--------|------------|
| **User input** | Fractional (units of lattice vectors) |
| **Internal storage** | Cartesian |
| `cell_vectors` | Cartesian |
| `displacements` in `add_hopping()` | Fractional |
| `reciprocal_vectors()` | Cartesian |

### Energy Units

All energies are **dimensionless** in units of hopping parameter `|t|`. Default: `t = -1.0`.

- Kagome eigenvalues: -2|t| to +4|t|
- Flat band at E = -2|t|

---

## CPU/GPU Split for Scientific Computing

**IMPORTANT**: This is a scientific calculation package. NOT everything runs on GPU.

### GPU (Compute-Intensive Operations)
- Bayesian optimization (LEVEL 7)
- CNN/self-attention impurity solver (LEVEL 4, future)
- Bayesian self-energy generation (LEVEL 4)
- Large matrix diagonalization (LEVEL 3)
- QGT computations with autograd (LEVEL 5)
- Dense tensor operations (all levels)
- Lorentzian broadening for DOS (LEVEL 5)

### CPU (Control Logic & I/O)
- DMFT loop control logic (LEVEL 4)
- Plotting with matplotlib (LEVEL 5) - requires `.cpu()` or `.numpy()`
- File I/O (all levels)
- Loop overhead (all levels)
- Result storage (all levels)
- Parameter sweep control (LEVEL 7)

### User-Controlled Device Selection

**Design pattern: Allow users to specify device per epoch/iteration**

```python
def dmft_loop(Hk, omega, solver, max_iter=100, tol=1e-6,
              device_per_iter=None, verbose=False):
    """
    Args:
        device_per_iter: Optional callable or list mapping iteration -> device
            - None: Auto-detect once, use for all iterations (default)
            - "cpu": Force CPU for all iterations
            - "cuda": Force GPU for all iterations
            - callable: Function called each iteration: device = device_per_iter(iter)
            - list: Pre-specified devices per iteration: ["cpu", "cuda", "cpu", ...]
    """
    default_device = get_device()

    for i in range(max_iter):
        if device_per_iter is None:
            device = default_device
        elif callable(device_per_iter):
            device = device_per_iter(i)
        elif isinstance(device_per_iter, list):
            device = device_per_iter[i] if i < len(device_per_iter) else default_device
        else:
            device = default_device

        # Transfer to selected device, compute, bring back to CPU
        Hk_curr = Hk.to(device)
        # ... compute on device ...
        result = result.cpu()  # Back to CPU for control logic
```

---

## Development Rules

### Rule 1: Virtual Environment Setup (REQUIRED)

```bash
python -m venv env_condmatTensor
source env_condmatTensor/bin/activate  # Linux/Mac
# env_condmatTensor\Scripts\activate  # Windows

# CRITICAL: Install PyTorch FIRST via CUDA-specific URL
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### Rule 2: Test-After-Module with Kagome Examples

After completing each module (LEVEL), run Kagome examples to verify correctness:

```python
# After finishing a module, test with:
from condmatTensor.lattice import BravaisLattice, generate_kmesh, generate_k_path
from condmatTensor.core import BaseTensor
from condmatTensor.analysis import DOSCalculator, BandStructure
from condmatTensor.solvers import diagonalize

lattice, Hr = build_kagome_lattice()
k_path = generate_k_path(lattice, ['G', 'K', 'M', 'G'], 100)
Hk_path = Hr.to_k_space(k_path)
E_path, U_path = diagonalize(Hk_path)

# Verify: Flat band at ε = -2t
# Verify: Dirac points at K
```

### Rule 3: No Circular Dependencies
- Lower levels (1-3) never depend on higher levels (4-10)
- Each level can only depend on levels with smaller numbers

### Rule 4: Use BaseTensor for All Physics Objects
- Hamiltonians: `labels=['k', 'orb_i', 'orb_j']` or `['R', 'orb_i', 'orb_j']`
- Green's functions: `labels=['k', 'orb_i', 'orb_j', 'iwn']`
- Self-energies: `labels=['iwn']` (pure imaginary)

### Rule 5: Many-Body Algorithm Reference Discussion

Before implementing many-body algorithms:
1. Find primary reference paper (< 5 years old preferred)
2. Identify benchmark test cases
3. Extract formalism and equations
4. Create test case with known analytic result
5. Document in code comments with citation

**Algorithms requiring discussion:**
- DMFT self-consistency loop
- IPT solver
- QGT computation (autograd and analytic)
- Chern number / Z₂ invariant
- RGF transport
- ED with CNN-selected CI

### Rule 6: Import Order

Always import from lower levels first:

```python
# GOOD: Lower levels first
from condmatTensor.core import BaseTensor
from condmatTensor.lattice import BravaisLattice
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure

# BAD: Higher levels before lower
from condmatTensor.analysis import BandStructure  # LEVEL 5
from condmatTensor.core import BaseTensor         # LEVEL 1
```

---

## Import Patterns

```python
# LEVEL 1: Core
from condmatTensor.core import BaseTensor, get_device

# LEVEL 2: Lattice
from condmatTensor.lattice import (
    BravaisLattice, HoppingModel, generate_kmesh, generate_k_path
)

# LEVEL 3: Solvers
from condmatTensor.solvers import diagonalize

# LEVEL 4: Many-Body
from condmatTensor.manybody.preprocessing import (
    generate_matsubara_frequencies, BareGreensFunction, SelfEnergy, SpectralFunction
)
from condmatTensor.manybody.magnetic import KondoLatticeSolver, SpinFermionModel

# LEVEL 5: Analysis
from condmatTensor.analysis import DOSCalculator, ProjectedDOS, BandStructure

# LEVEL 7: Optimization
from condmatTensor.optimization import BayesianOptimizer, EffectiveArrayOptimizer
```

---

## Common Workflows

### Band Structure Calculation
```python
from condmatTensor.lattice import BravaisLattice, HoppingModel, generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure

lattice = BravaisLattice(cell_vectors, basis_positions, num_orbitals)
model = HoppingModel(lattice, orbital_labels=['A', 'B', 'C'])
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
    generate_matsubara_frequencies, BareGreensFunction, SelfEnergy
)

omega = generate_matsubara_frequencies(beta=10.0, n_max=128)
G0 = BareGreensFunction.from_hamiltonian(Hk, omega, mu=0.0)
Sigma = SelfEnergy(omega, 0.0)
G = G0.apply_self_energy(Sigma)
```

### Bayesian Optimization
```python
from condmatTensor.optimization import BayesianOptimizer

opt = BayesianOptimizer(bounds=[(0, 1), (0, 1)], backend="auto")
X_best, y_best = opt.optimize(objective_fn, n_init=10, n_iter=50, device=device)
```

---

## Bayesian Optimization Backend Priority

| Priority | Backend | Package | Description |
|----------|---------|---------|-------------|
| **1** | SOBER | `sober-bo==2.0.4` | Sequential Optimization using Ensemble of Regressors |
| **2** | BoTorch | `botorch>=0.9.0` | Gaussian Process with Expected Improvement |
| **3** | Simple | Python stdlib | Thompson sampling / random search |

---

## Validation Examples

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

---

## Key Physics Validation Points

### Kagome Lattice
- **Flat band**: Expected at E = -2|t|
- **Dirac points**: Expected at K points in Brillouin zone
- **Band range**: -2|t| to +4|t|

### Kagome-F (4 sites/cell)
- **Total bands**: 8 (4 sites × 2 spin)
- **f-d hybridization**: Expected between f and d orbitals
- **Effective model**: Bayesian downfolding to 6 bands

---

## Documentation Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Claude Code development guide |
| `plans/architecture_plan.md` | Comprehensive architecture reference |
| `plans/DEPENDENCY_ANALYSIS.md` | Comparison with NumPy/TRIQS/WannierTools |

---

## Design Principles

1. **Tensor-First**: All physics data in `BaseTensor`
2. **Unified Interface**: H, G, Σ all use same class
3. **GPU-Ready**: User-controlled device selection per epoch
4. **Clear Dependencies**: Explicit level hierarchy
5. **Layered Architecture**: No circular imports
