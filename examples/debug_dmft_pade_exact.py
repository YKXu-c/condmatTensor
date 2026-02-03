"""Debug DMFT Pade issue - exact reproduction of kagome_f_dmft.py setup."""

import torch
import math
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

from condmatTensor.manybody.preprocessing import (
    SpectralFunction,
    generate_matsubara_frequencies,
)
from condmatTensor.core import BaseTensor

device = torch.device('cpu')

# Exact parameters from kagome_f_dmft.py
beta = 10.0
n_max_param = 32  # This is the parameter used for frequency selection
U_d = 0.5  # d-orbital U
U_f = 4.0  # f-orbital U

print("="*70)
print("Debugging DMFT Pade Continuation - Exact kagome_f_dmft.py Setup")
print("="*70)

# Generate Matsubara frequencies (same as kagome_f_dmft.py)
omega = generate_matsubara_frequencies(beta=beta, n_max=64)
n_iwn = len(omega)
print(f"\nMatsubara frequency grid:")
print(f"  n_iwn: {n_iwn}")
print(f"  Range: [{omega[0]}, {omega[-1]}]")

# Create G_int exactly as kagome_f_dmft.py does it
# G_int is the local Green's function from DMFT
n_orb = 4
G_int = torch.zeros((n_iwn, n_orb, n_orb), dtype=torch.complex128, device=device)

# Build the Green's function with self-energy (simulating DMFT output)
# For each orbital i: G_ii(iω) = 1/(iω - ε_i - Σ_ii(iω))
for orb in range(n_orb):
    # Orbital energies from Kagome-F model
    epsilon = -1.0 + orb * 0.5  # [-1.0, -0.5, 0, 0.5]
    U_val = U_f if orb == 3 else U_d  # f-orbital has larger U

    for i in range(n_iwn):
        # IPT-like self-energy: Σ(iω) = U²/(iω)
        if i != n_iwn // 2:  # Avoid iω = 0
            sigma = U_val**2 / omega[i]
        else:
            sigma = 0.0

        # Green's function
        denom = omega[i] - epsilon - sigma
        if abs(denom) > 1e-12:
            G_int[i, orb, orb] = 1.0 / denom

print(f"\nG_int shape: {G_int.shape}")
print(f"G_int dtype: {G_int.dtype}")

# Print sample values for orbital 0
print(f"\nOrbital 0 G values (first 5 positive frequencies):")
for i in range(64, 69):
    print(f"  n={i-64}: G[{i},0,0] = {G_int[i,0,0]}")

# Create BaseTensor
G_int_base = BaseTensor(
    tensor=G_int,
    labels=['iwn', 'orb_i', 'orb_j'],
)

# Setup spectral function
spectral = SpectralFunction()

# Use the same frequency range as kagome_f_dmft.py
# omega_min, omega_max = calculate_dos_range(...) gives [-20.75, 18.67]
omega_min = -20.75
omega_max = 18.67
omega_pade = torch.linspace(omega_min, omega_max, 1000, device=device)
eta_pade = 0.05

print(f"\n{'='*70}")
print(f"Running Pade continuation (same parameters as kagome_f_dmft.py)...")
print(f"{'='*70}")
print(f"  omega range: [{omega_min}, {omega_max}]")
print(f"  n_omega: {len(omega_pade)}")
print(f"  eta: {eta_pade}")
print(f"  beta: {beta}")
print(f"  n_max: {n_max_param}")

try:
    # This is the exact call from kagome_f_dmft.py line 793-796
    _, A_int_pade = spectral.from_matsubara(
        G_int_base, omega_pade, eta=eta_pade, method="pade",
        beta=beta, n_min=0, n_max=n_max_param
    )

    dos_int_pade = spectral.compute_dos(A_int_pade)

    print(f"\n{'='*70}")
    print(f"Pade continuation completed!")
    print(f"{'='*70}")
    print(f"  A_int_pade shape: {A_int_pade.shape}")
    print(f"  A_int_pade range: [{A_int_pade.min():.6f}, {A_int_pade.max():.6f}]")
    print(f"  A_int_pade mean: {A_int_pade.mean():.6f}")

    print(f"\n  dos_int_pade shape: {dos_int_pade.shape}")
    print(f"  dos_int_pade range: [{dos_int_pade.min():.6f}, {dos_int_pade.max():.6f}]")
    print(f"  dos_int_pade mean: {dos_int_pade.mean():.6f}")

    # Check individual orbitals
    print(f"\n  Individual orbitals:")
    for orb in range(n_orb):
        print(f"    Orbital {orb}: [{A_int_pade[:, orb].min():.6f}, {A_int_pade[:, orb].max():.6f}], mean={A_int_pade[:, orb].mean():.6f}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Individual orbitals
    colors = ['blue', 'red', 'green', 'purple']
    for orb in range(n_orb):
        axes[0].plot(omega_pade.cpu(), A_int_pade[:, orb].cpu(),
                    label=f'Orbital {orb}', color=colors[orb], linewidth=1.5)
    axes[0].set_xlabel(r"Energy $\omega$ ($|t|$)")
    axes[0].set_ylabel("Spectral Function $A(\\omega)$")
    axes[0].set_title("DMFT Pade - Individual Orbitals")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Total DOS
    axes[1].plot(omega_pade.cpu(), dos_int_pade.cpu(),
                color='darkgreen', linewidth=2, label='Pade DOS')
    axes[1].set_xlabel(r"Energy $\omega$ ($|t|$)")
    axes[1].set_ylabel("DOS")
    axes[1].set_title("DMFT Pade - Total DOS")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('debug_dmft_pade_exact.png', dpi=100)
    print(f"\nSaved plot to: debug_dmft_pade_exact.png")

    # Diagnosis
    print(f"\n{'='*70}")
    print(f"Diagnosis:")
    print(f"{'='*70}")

    if dos_int_pade.mean() < 0.01:
        print("❌ CRITICAL: DOS is nearly zero (mean < 0.01)")
        print("   This is the bug we see in kagome_f_dmft_dos_pade.png!")
        print("   The Pade continuation is producing incorrect results.")
    elif dos_int_pade.mean() < 0.1:
        print("⚠️  WARNING: DOS is very small (mean < 0.1)")
        print("   The Pade continuation may have issues.")
    elif dos_int_pade.mean() < 1.0:
        print("⚠️  CAUTION: DOS is small (mean < 1.0)")
    else:
        print("✓ OK: DOS has reasonable magnitude")

    # Check for peaks
    peak_threshold = 1.0
    n_peaks = (dos_int_pade > peak_threshold).sum().item()
    print(f"\nPeak analysis:")
    print(f"  Points above {peak_threshold}: {n_peaks} out of {len(dos_int_pade)}")

    if n_peaks < 10:
        print("  ❌ Very few or no peaks - this indicates a problem!")
    else:
        print(f"  ✓ Has {n_peaks} points above threshold")

except Exception as e:
    print(f"\n❌ Pade continuation FAILED:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
