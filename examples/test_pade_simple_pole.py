"""Test Pade continuation with a simple pole to verify it works correctly."""

import torch
import math
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

from condmatTensor.manybody.preprocessing import SpectralFunction
from condmatTensor.core import BaseTensor

device = torch.device('cpu')
beta = 10.0

print("="*60)
print("Testing Pade Continuation with Simple Pole")
print("="*60)

# Create a simple Green's function: G(iωₙ) = 1/(iωₙ - ε)
# where ε = -1.0
n_iwn = 129
n_vals = torch.arange(n_iwn, device=device) - n_iwn // 2
omega_n = (2 * n_vals + 1) * math.pi / beta
z_iwn_all = 1j * omega_n

# Single orbital with simple pole
epsilon = -1.0
G_simple = torch.zeros((n_iwn, 1, 1), dtype=torch.complex128, device=device)
for i in range(n_iwn):
    G_simple[i, 0, 0] = 1.0 / (z_iwn_all[i] - epsilon)

# Create BaseTensor
G_base = BaseTensor(tensor=G_simple, labels=['iwn', 'orb_i', 'orb_j'])
spectral = SpectralFunction()

# Real frequency grid
omega_real = torch.linspace(-3, 3, 100, device=device)
eta = 0.05

print(f"\nGreen's function: G(iωₙ) = 1/(iωₙ - ε)")
print(f"  ε = {epsilon}")
print(f"  beta = {beta}")

print(f"\nRunning Pade continuation...")
print(f"  omega range: [{omega_real[0]}, {omega_real[-1]}]")
print(f"  n_omega: {len(omega_real)}")
print(f"  eta: {eta}")
print(f"  n_max: 32")

try:
    omega_out, A_pade = spectral.from_matsubara(
        G_base, omega_real, eta=eta, method="pade",
        beta=beta, n_min=0, n_max=32
    )

    print(f"\n✓ Pade continuation successful!")
    print(f"  Output shape: {A_pade.shape}")
    print(f"  A_pade range: [{A_pade.min():.6f}, {A_pade.max():.6f}]")

    # Compare with analytic
    A_analytic = torch.zeros_like(omega_real)
    gamma = eta
    for i, w in enumerate(omega_real):
        A_analytic[i] = (gamma / math.pi) / ((w - epsilon)**2 + gamma**2)

    print(f"\nAnalytic solution:")
    print(f"  A_analytic range: [{A_analytic.min():.6f}, {A_analytic.max():.6f}]")

    l2_error = torch.sqrt(torch.mean((A_pade[:, 0] - A_analytic)**2))
    print(f"\nL2 error: {l2_error:.10f}")

    if l2_error < 0.01:
        print("✓ EXCELLENT: Pade continuation is very accurate!")
    elif l2_error < 0.1:
        print("✓ GOOD: Pade continuation is reasonably accurate")
    else:
        print("❌ ERROR: Pade continuation has large errors!")

    # Plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(omega_real.cpu(), A_analytic.cpu(),
            label='Analytic', linewidth=3, color='black', linestyle='--')
    ax.plot(omega_real.cpu(), A_pade[:, 0].cpu(),
            label='Pade', linewidth=2, color='red')

    ax.set_xlabel(r"Energy $\omega$ ($|t|$)")
    ax.set_ylabel("Spectral Function $A(\\omega)$")
    ax.set_title(f"Pade Continuation Test: Simple Pole at ε={epsilon}\nL2 Error = {l2_error:.6f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_pade_simple_pole.png', dpi=100)
    print(f"\nSaved plot to: test_pade_simple_pole.png")

except Exception as e:
    print(f"\n❌ Pade continuation FAILED:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
