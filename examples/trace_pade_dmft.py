"""Trace the exact data flow in Pade continuation for DMFT."""

import torch
import math
import sys
sys.path.insert(0, 'src')

from condmatTensor.manybody.analytic_continuation import PadeContinuation

device = torch.device('cpu')
beta = 10.0

# Simulate a simple DMFT Green's function for ONE orbital
n_iwn = 129
n_vals = torch.arange(n_iwn, device=device) - n_iwn // 2
omega_n = (2 * n_vals + 1) * math.pi / beta
z_iwn_all = 1j * omega_n

# Create G for orbital 0 (ε = -1.0, weak interaction)
G_orb0 = torch.zeros(n_iwn, dtype=torch.complex128, device=device)
epsilon = -1.0
U = 0.5

for i in range(n_iwn):
    if i != n_iwn // 2:
        sigma = U**2 / z_iwn_all[i]
    else:
        sigma = 0.0
    denom = z_iwn_all[i] - epsilon - sigma
    G_orb0[i] = 1.0 / denom if abs(denom) > 1e-12 else 0.0

print("="*60)
print("Tracing Pade Continuation for Orbital 0")
print("="*60)

# Frequency selection (as done in preprocessing.py)
n_max = 32
n_min = 0
idx_start = (n_iwn // 2) + n_min  # = 64
idx_end_candidate = min(idx_start + n_max, n_iwn)  # = 96

print(f"\nFrequency selection:")
print(f"  n_iwn: {n_iwn}")
print(f"  n_min: {n_min}, n_max: {n_max}")
print(f"  idx_start: {idx_start}, idx_end_candidate: {idx_end_candidate}")

# Adaptive frequency selection
G_candidate = G_orb0[idx_start:idx_end_candidate]
G_mag = torch.abs(G_candidate)
print(f"  G_candidate shape: {G_candidate.shape}")
print(f"  |G| range: [{G_mag.min():.6f}, {G_mag.max():.6f}]")

tolerance = 1e-12
relative_threshold = 1e-3 * G_mag.max().item()
threshold = max(relative_threshold, tolerance)
print(f"  Threshold: {threshold:.6e}")

significant_mask = G_mag > threshold
n_significant = significant_mask.sum().item()
print(f"  Significant frequencies: {n_significant}")

n_optimal = min(max(n_significant, 8), 30)
print(f"  n_optimal: {n_optimal}")

idx_end = idx_start + n_optimal

# Extract the selected G values and frequencies
G_selected = G_orb0[idx_start:idx_end]
n_vals_selected = n_vals[idx_start:idx_end]
wn = (2 * n_vals_selected + 1) * math.pi / beta
z_iwn = 1j * wn

print(f"\nSelected data:")
print(f"  G_selected shape: {G_selected.shape}")
print(f"  n_vals: {n_vals_selected.tolist()[:5]}...{n_vals_selected.tolist()[-5:]}")
print(f"  z_iwn: {z_iwn[:3]}...")

# First few G values
print(f"\nFirst few G values:")
for i in range(min(5, len(G_selected))):
    print(f"  i={i}, n={n_vals_selected[i]}: z={z_iwn[i]:.4f}, G={G_selected[i]}")

# Now test Pade directly
pade = PadeContinuation(tolerance=1e-12)

# Real frequency grid (narrower range for debugging)
omega_real = torch.linspace(-3, 3, 100, device=device)
eta = 0.05

print(f"\n{'='*60}")
print(f"Running Pade continuation...")
print(f"  omega range: [{omega_real[0]}, {omega_real[-1]}]")
print(f"  n_omega: {len(omega_real)}")

A_pade = pade._pade_continued_fraction(
    G_selected, z_iwn, omega_real, eta, device
)

print(f"\nResults:")
print(f"  A_pade shape: {A_pade.shape}")
print(f"  A_pade range: [{A_pade.min():.6f}, {A_pade.max():.6f}]")
print(f"  A_pade mean: {A_pade.mean():.6f}")

# Compare with analytic
epsilon_analytic = -1.0
gamma = eta
A_analytic = torch.zeros_like(omega_real)
for i, w in enumerate(omega_real):
    A_analytic[i] = (gamma / math.pi) / ((w - epsilon_analytic)**2 + gamma**2)

print(f"\nAnalytic comparison:")
print(f"  A_analytic range: [{A_analytic.min():.6f}, {A_analytic.max():.6f}]")
print(f"  L2 error: {torch.sqrt(torch.mean((A_pade - A_analytic)**2)):.6f}")

# Check if peak is at right location
peak_idx = A_pade.argmax()
peak_omega = omega_real[peak_idx].item()
print(f"  Peak at ω = {peak_omega:.3f} (expected near {epsilon_analytic})")

# Print some sample values
print(f"\nSample values:")
for i in [20, 40, 50, 60, 80]:
    w = omega_real[i].item()
    a = A_pade[i].item()
    a_ref = A_analytic[i].item()
    print(f"  ω = {w:6.2f}: A_pade = {a:.6f}, A_analytic = {a_ref:.6f}")

# Now test with the wider frequency range (as in kagome_f_dmft.py)
print(f"\n{'='*60}")
print(f"Testing with wider frequency range (as in kagome_f_dmft.py)...")
print(f"{'='*60}")

omega_wide = torch.linspace(-20.75, 18.67, 1000, device=device)
A_wide = pade._pade_continued_fraction(
    G_selected, z_iwn, omega_wide, eta, device
)

print(f"  A_wide shape: {A_wide.shape}")
print(f"  A_wide range: [{A_wide.min():.6f}, {A_wide.max():.6f}]")
print(f"  A_wide mean: {A_wide.mean():.6f}")

# Check peak location
peak_idx_wide = A_wide.argmax()
peak_omega_wide = omega_wide[peak_idx_wide].item()
print(f"  Peak at ω = {peak_omega_wide:.3f} (expected near {epsilon_analytic})")

# Sample values near the expected peak
print(f"\nSample values around expected peak:")
peak_center = 50  # Index where omega ≈ -1.0 in the wide range
for i in range(peak_center - 5, peak_center + 6):
    w = omega_wide[i].item()
    a = A_wide[i].item()
    if abs(w - epsilon_analytic) < 0.1:
        print(f"  ω = {w:7.2f}: A_wide = {a:.6f}  ← near peak")
    else:
        print(f"  ω = {w:7.2f}: A_wide = {a:.6f}")
