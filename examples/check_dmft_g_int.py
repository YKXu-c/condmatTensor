"""Check the actual G_int values from DMFT to debug the Pade issue."""

import torch
import math
import sys
sys.path.insert(0, 'src')

from condmatTensor.manybody.preprocessing import generate_matsubara_frequencies
from condmatTensor.core import BaseTensor

device = torch.device('cpu')
beta = 10.0

# Generate Matsubara frequencies
omega = generate_matsubara_frequencies(beta=beta, n_max=64)
n_iwn = len(omega)

print("="*70)
print("Checking DMFT G_int Values")
print("="*70)

# Simulate G_int as it would come from DMFT loop
# This simulates the output of SingleSiteDMFTLoop

# The DMFT loop computes: G_loc(k,iω) = [iω + μ - H(k) - Σ(iω)]^(-1)
# Then: G_loc(iω) = (1/N_k) Σ_k G_loc(k,iω)

# For testing, create a realistic G_int with multi-orbital structure
n_orb = 4
G_int = torch.zeros((n_iwn, n_orb, n_orb), dtype=torch.complex128, device=device)

# Use parameters from kagome_f_dmft.py
U_d = 0.5
U_f = 4.0

# Simulate G_int with self-energy from IPT solver
for orb in range(n_orb):
    # Kagome-F orbital energies
    epsilon = -1.0 + orb * 0.5
    U_val = U_f if orb == 3 else U_d

    for i in range(n_iwn):
        # IPT self-energy: Σ(iω) = U²/(iω)
        if i != n_iwn // 2:
            sigma = U_val**2 / omega[i]
        else:
            sigma = 0.0

        # Green's function: G = 1/(iω - ε - Σ)
        denom = omega[i] - epsilon - sigma
        if abs(denom) > 1e-12:
            G_int[i, orb, orb] = 1.0 / denom

print(f"\nG_int shape: {G_int.shape}")
print(f"G_int dtype: {G_int.dtype}")

# Check the values for each orbital
print(f"\nPer-orbital G values at low frequencies:")
print(f"{'Orbital':<10} {'G[64] (iω₀)':<30} {'G[65] (iω₁)':<30} {'|G|_max':<15} {'|G|_min':<15}")
print("="*100)

for orb in range(n_orb):
    epsilon = -1.0 + orb * 0.5
    G_at_0 = G_int[64, orb, orb]
    G_at_1 = G_int[65, orb, orb]
    G_max = torch.abs(G_int[64:, orb, orb]).max()
    G_min = torch.abs(G_int[64:, orb, orb]).min()

    print(f"  {orb:<8} {str(G_at_0):<30} {str(G_at_1):<30} {G_max:<15.6f} {G_min:<15.6f}")

# Now test Pade continuation
from condmatTensor.manybody.preprocessing import SpectralFunction

spectral = SpectralFunction()

# Use same parameters as kagome_f_dmft.py
omega_min, omega_max = -20.75, 18.67
omega_pade = torch.linspace(omega_min, omega_max, 1000, device=device)
eta_pade = 0.05

print(f"\n{'='*70}")
print(f"Testing Pade continuation")
print(f"{'='*70}")
print(f"  omega range: [{omega_min}, {omega_max}]")
print(f"  n_omega: {len(omega_pade)}")
print(f"  eta: {eta_pade}")
print(f"  beta: {beta}")
print(f"  n_max: 32")

G_int_base = BaseTensor(
    tensor=G_int,
    labels=['iwn', 'orb_i', 'orb_j'],
)

_, A_int_pade = spectral.from_matsubara(
    G_int_base, omega_pade, eta=eta_pade, method="pade",
    beta=beta, n_min=0, n_max=32
)

dos_int_pade = spectral.compute_dos(A_int_pade)

print(f"\n{'='*70}")
print(f"Results")
print(f"{'='*70}")
print(f"  A_int_pade shape: {A_int_pade.shape}")
print(f"  A_int_pade range: [{A_int_pade.min():.6f}, {A_int_pade.max():.6f}]")
print(f"  A_int_pade mean: {A_int_pade.mean():.6f}")

print(f"\n  dos_int_pade shape: {dos_int_pade.shape}")
print(f"  dos_int_pade range: [{dos_int_pade.min():.6f}, {dos_int_pade.max():.6f}]")
print(f"  dos_int_pade mean: {dos_int_pade.mean():.6f}")

print(f"\n  Per-orbital statistics:")
for orb in range(n_orb):
    print(f"    Orbital {orb}: max={A_int_pade[:, orb].max():.4f}, mean={A_int_pade[:, orb].mean():.6f}")

# Check if the issue is present
is_flat = dos_int_pade.mean() < 0.1
print(f"\n{'='*70}")
print(f"Diagnosis: {'FLAT (mean < 0.1)' if is_flat else 'OK (mean >= 0.1)'}")
print(f"{'='*70}")

# Check specific energy regions
print(f"\nSpectral weight distribution:")
print(f"  Energy region            DOS mean        DOS max")
print(f"  '-' * 50")

regions = [
    (-20.75, -10, "Far left"),
    (-10, -5, "Left Hubbard band"),
    (-5, 0, "Left quasiparticle"),
    (0, 5, "Right quasiparticle"),
    (5, 10, "Right Hubbard band"),
    (10, 18.67, "Far right"),
]

for w_min, w_max, label in regions:
    mask = (omega_pade >= w_min) & (omega_pade <= w_max)
    if mask.sum() > 0:
        dos_region = dos_int_pade[mask]
        print(f"  {label:<25} {dos_region.mean():<15.6f} {dos_region.max():<15.6f}")
