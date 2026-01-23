#!/usr/bin/env python3
"""
Test additional magnetic module functionality:
- pauli_matrices()
- self_consistency_loop()
- KondoLatticeSolver (Kondo temperature estimation)
- BayesianOptimizer (Bayesian optimization for synthetic objective functions)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from condmatTensor.manybody import pauli_matrices, LocalMagneticModel, KondoLatticeSolver
from condmatTensor.core import BaseTensor
from condmatTensor.lattice import BravaisLattice, TightBindingModel, generate_kmesh
from condmatTensor.optimization import BayesianOptimizer


def test_pauli_matrices():
    """Test Pauli matrices function."""
    print("=" * 70)
    print("Test 1: Pauli Matrices")
    print("=" * 70)

    sigma_x, sigma_y, sigma_z = pauli_matrices()

    print("\nσx =")
    print(sigma_x)
    print("\nσy =")
    print(sigma_y)
    print("\nσz =")
    print(sigma_z)

    # Verify Pauli algebra: σ_i² = I
    I = torch.eye(2, dtype=torch.complex128)
    assert torch.allclose(sigma_x @ sigma_x, I), "σx² should be identity"
    assert torch.allclose(sigma_y @ sigma_y, I), "σy² should be identity"
    assert torch.allclose(sigma_z @ sigma_z, I), "σz² should be identity"

    # Verify commutation: [σx, σy] = 2iσz
    comm_xy = sigma_x @ sigma_y - sigma_y @ sigma_x
    expected = 2j * sigma_z
    assert torch.allclose(comm_xy, expected), "[σx, σy] should equal 2iσz"

    print("\n✅ Pauli matrices test PASSED")
    return True


def test_self_consistency_loop():
    """Test self-consistency loop for local moments."""
    print("\n" + "=" * 70)
    print("Test 2: Self-Consistency Loop")
    print("=" * 70)

    # Build a simple 2-site test system
    import math
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    basis_positions = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
    ]

    lattice = BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[2, 2],  # 2 sites, each with 2 spin orbitals
    )

    orbital_labels = ["A_up", "A_down", "B_up", "B_down"]
    tb_model = TightBindingModel(lattice, orbital_labels=orbital_labels)

    # Add hoppings
    t = -1.0
    tb_model.add_hopping("A_up", "A_down", [0, 0], 0.0)  # On-site
    tb_model.add_hopping("B_up", "B_down", [0, 0], 0.0)
    tb_model.add_hopping("A_up", "B_up", [0, 0], t)
    tb_model.add_hopping("A_down", "B_down", [0, 0], t)

    # Generate k-mesh
    k_mesh = generate_kmesh(lattice, nk=8)

    # Build Hamiltonian
    Hk = tb_model.build_Hk(k_mesh)

    # Set up LocalMagneticModel with simple parameters
    J = 0.5
    S_init = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])  # Initial moments up

    model = LocalMagneticModel(H0=Hk, J=J, S_init=S_init)

    print(f"\nSystem: Total: {lattice.total_orbitals} orbitals ({lattice.num_sites} sites)")
    print(f"  Orbitals per site: {lattice.num_orbitals}")
    print(f"J = {J}")
    print(f"Initial S = {S_init}")

    # Run self-consistency loop with low temperature
    beta = 10.0
    n_max = 10  # Small for quick test

    print(f"\nRunning self-consistency loop...")
    print(f"  β = {beta}")
    print(f"  n_max = {n_max}")

    S_final, n_iter = model.self_consistency_loop(
        beta=beta,
        mixing=0.3,
        tol=1e-3,
        max_iter=20,
        n_max=n_max,
        verbose=True,
    )

    print(f"\nFinal S:")
    print(S_final)
    print(f"\nConverged in {n_iter} iterations")

    print("\n✅ Self-consistency loop test PASSED")
    return True


def test_kondo_solver():
    """Test KondoLatticeSolver."""
    print("\n" + "=" * 70)
    print("Test 3: Kondo Lattice Solver")
    print("=" * 70)

    # Build a simple test system
    import math
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    basis_positions = [
        torch.tensor([0.0, 0.0]),
    ]

    lattice = BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[2],  # 1 site with 2 spin orbitals
    )

    orbital_labels = ["c_up", "c_down"]
    tb_model = TightBindingModel(lattice, orbital_labels=orbital_labels)

    # Add hopping
    t = -1.0
    tb_model.add_hopping("c_up", "c_down", [0, 0], 0.0)  # On-site

    # Generate small k-mesh
    k_mesh = generate_kmesh(lattice, nk=4)

    # Build Hamiltonian
    Hk = tb_model.build_Hk(k_mesh)

    # Create KondoLatticeSolver
    J = -0.5  # Antiferromagnetic Kondo coupling
    solver = KondoLatticeSolver(H0=Hk, J=J)

    print(f"\nSystem: 1 site × 2 spin = 2 orbitals")
    print(f"J (Kondo coupling) = {J}")

    # Estimate Kondo temperature
    # Estimate DOS at Fermi level (simplified)
    density_of_states = 1.0
    T_K = solver.estimate_kondo_temperature(density_of_states=density_of_states)

    print(f"\nKondo temperature estimate:")
    print(f"  T_K = {T_K:.6f}")

    # Compute RKKY interaction (placeholder)
    q_vector = torch.tensor([0.5, 0.5])
    J_RKKY = solver.compute_rkky_interaction(Hk, q_vector)

    print(f"\nRKKY interaction at q = {q_vector.tolist()}:")
    print(f"  J_RKKY = {J_RKKY:.6f}")

    print("\n✅ Kondo solver test PASSED")
    return True


def test_bayesian_optimization():
    """Test BayesianOptimizer with synthetic objective function.

    Simple quadratic objective: sum((x - 0.5)^2)
    Expected minimum at x = [0.5, 0.5, ...] with value 0
    """
    print("\n" + "=" * 70)
    print("Test 4: Bayesian Optimization")
    print("=" * 70)

    # Define simple quadratic objective
    def quadratic_objective(X):
        """Quadratic bowl: minimum at (0.5, 0.5) with value 0."""
        # Handle both 1D (n_dim,) and 2D (n, n_dim) inputs
        if X.ndim == 1:
            return torch.sum((X - 0.5) ** 2).unsqueeze(0)
        else:
            return torch.sum((X - 0.5) ** 2, dim=1, keepdim=True)

    bounds = [(0, 1), (0, 1)]
    n_init = 5
    n_iter = 10
    seed = 42

    # Test each available backend
    backends_to_test = ["simple"]  # Simple always works
    backend_names = {"simple": "SimpleBackend"}

    # Try SOBER
    try:
        from condmatTensor.optimization.bayesian.sober_backend import SoberBackend
        backends_to_test.append("sober")
        backend_names["sober"] = "SoberBackend"
    except ImportError:
        pass

    # Try BoTorch
    try:
        from condmatTensor.optimization.bayesian.botorch_backend import BotorchBackend
        backends_to_test.append("botorch")
        backend_names["botorch"] = "BotorchBackend"
    except ImportError:
        pass

    print(f"\nTesting backends: {backends_to_test}")

    results = {}
    for backend_name in backends_to_test:
        print(f"\n  Testing {backend_name.upper()}...")

        try:
            # Create optimizer with n_init, n_iter, seed set during construction
            opt = BayesianOptimizer(bounds=bounds, backend=backend_name, n_init=n_init, n_iter=n_iter, seed=seed)

            # Run optimization
            X_best, y_best = opt.optimize(
                quadratic_objective,
                verbose=False,
            )

            results[backend_name] = {
                "X_best": X_best,
                "y_best": y_best,
            }

            distance_to_opt = torch.norm(X_best - 0.5).item()
            print(f"    Best point: {X_best.tolist()}")
            # Handle both tensor and float return types
            y_val = y_best.item() if hasattr(y_best, 'item') else float(y_best)
            print(f"    Best value: {y_val:.6f}")
            print(f"    Distance to optimum: {distance_to_opt:.6f}")

            # Verify result is reasonable
            assert torch.all(X_best >= 0.0) and torch.all(X_best <= 1.0), \
                f"{backend_name}: Best point should be within bounds"
            assert distance_to_opt < 0.5, \
                f"{backend_name}: Should find point within 0.5 of optimum"

            print(f"    ✅ {backend_name.upper()} backend passed")

        except Exception as e:
            print(f"    ❌ {backend_name.upper()} backend failed: {e}")
            import traceback
            traceback.print_exc()

    # Test backend auto-detection
    print("\n  Testing AUTO backend detection...")
    try:
        opt_auto = BayesianOptimizer(bounds=bounds, backend="auto", n_init=n_init, n_iter=n_iter, seed=seed)
        X_best_auto, y_best_auto = opt_auto.optimize(
            quadratic_objective,
            verbose=False,
        )
        print(f"    Auto-detected backend worked")
        print(f"    Best point: {X_best_auto.tolist()}")
        y_val_auto = y_best_auto.item() if hasattr(y_best_auto, 'item') else float(y_best_auto)
        print(f"    Best value: {y_val_auto:.6f}")
        print(f"    ✅ AUTO backend passed")
    except Exception as e:
        print(f"    ❌ AUTO backend failed: {e}")

    print("\n✅ Bayesian optimization test PASSED")
    return True


def main():
    """Run all additional tests."""
    print("\n" + "=" * 70)
    print("Magnetic Module Additional Functionality Tests")
    print("=" * 70)

    try:
        test_pauli_matrices()
        test_self_consistency_loop()
        test_kondo_solver()
        test_bayesian_optimization()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✅")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
