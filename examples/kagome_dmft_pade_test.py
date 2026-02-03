"""Comprehensive test for Pade analytic continuation.

Tests the Pade continuation module with various Green's function types:
1. Simple pole (single orbital)
2. Bethe lattice (semi-elliptical DOS)
3. Multi-orbital system
4. DMFT-like Green's function with self-energy

This validates that the analytic_continuation module is working correctly.
"""

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
beta = 10.0

print("="*70)
print("Pade Analytic Continuation - Comprehensive Test Suite")
print("="*70)

# ========================================================================
# Test 1: Simple pole (G = 1/(iω - ε))
# ========================================================================
print("\n" + "="*70)
print("TEST 1: Simple Pole")
print("="*70)

omega = generate_matsubara_frequencies(beta=beta, n_max=64)
epsilon = -1.0

G_simple = torch.zeros((len(omega), 1, 1), dtype=torch.complex128, device=device)
for i, iwn in enumerate(omega):
    G_simple[i, 0, 0] = 1.0 / (iwn - epsilon)

G_base = BaseTensor(tensor=G_simple, labels=['iwn', 'orb_i', 'orb_j'])
spectral = SpectralFunction()

omega_real = torch.linspace(-3, 3, 100, device=device)
omega_out, A_pade = spectral.from_matsubara(
    G_base, omega_real, eta=0.05, method="pade",
    beta=beta, n_min=0, n_max=32
)

# Compare with analytic
A_analytic = torch.zeros_like(omega_real)
for i, w in enumerate(omega_real):
    A_analytic[i] = (0.05 / math.pi) / ((w - epsilon)**2 + 0.05**2)

l2_error_1 = torch.sqrt(torch.mean((A_pade[:, 0] - A_analytic)**2)).item()
print(f"  L2 Error: {l2_error_1:.10f}")
print(f"  Result: {'✓ PASS' if l2_error_1 < 0.01 else '✗ FAIL'}")

# ========================================================================
# Test 2: Bethe lattice (semi-elliptical DOS)
# ========================================================================
print("\n" + "="*70)
print("TEST 2: Bethe Lattice")
print("="*70)

# Bethe lattice Green's function: G(iω) = (2/D²)(ω - sqrt(ω² - D²))
# with D = 4 (half-bandwidth)
D = 4.0
t = 1.0
G_bethe = torch.zeros((len(omega), 1, 1), dtype=torch.complex128, device=device)

for i, iwn in enumerate(omega):
    # Bethe lattice Green's function on Matsubara frequencies
    sqrt_arg = iwn**2 - D**2
    # Branch cut: sqrt(z² - D²) with Im(sqrt) > 0 for Im(z) > 0
    sqrt_val = torch.sqrt(sqrt_arg + 0j)
    G_bethe[i, 0, 0] = (2 / D**2) * (iwn - sqrt_val)

G_base_bethe = BaseTensor(tensor=G_bethe, labels=['iwn', 'orb_i', 'orb_j'])

omega_bethe = torch.linspace(-5, 5, 100, device=device)
omega_out, A_pade_bethe = spectral.from_matsubara(
    G_base_bethe, omega_bethe, eta=0.05, method="pade",
    beta=beta, n_min=0, n_max=32
)

# Semi-elliptical DOS for comparison
A_bethe_analytic = torch.zeros_like(omega_bethe)
for i, w in enumerate(omega_bethe):
    if abs(w) < D:
        A_bethe_analytic[i] = (2 / (math.pi * D**2)) * math.sqrt(D**2 - w**2)
    else:
        A_bethe_analytic[i] = 0.0

# Convolve with Lorentzian for finite eta
A_bethe_broadened = torch.zeros_like(omega_bethe)
eta = 0.05
for i, w in enumerate(omega_bethe):
    for j, w_ref in enumerate(omega_bethe):
        if abs(w_ref) < D:
            lorentzian = (eta / math.pi) / ((w - w_ref)**2 + eta**2)
            A_bethe_broadened[i] += A_bethe_analytic[j] * lorentzian * (omega_bethe[1] - omega_bethe[0])

l2_error_2 = torch.sqrt(torch.mean((A_pade_bethe[:, 0] - A_bethe_broadened)**2)).item()
print(f"  L2 Error: {l2_error_2:.10f}")
print(f"  Peak value: Pade={A_pade_bethe[:, 0].max():.4f}, Analytic≈{A_bethe_broadened.max():.4f}")
print(f"  Result: {'✓ PASS' if l2_error_2 < 0.1 else '✗ FAIL'}")

# ========================================================================
# Test 3: Multi-orbital system
# ========================================================================
print("\n" + "="*70)
print("TEST 3: Multi-Orbital System")
print("="*70)

n_orb = 4
G_multi = torch.zeros((len(omega), n_orb, n_orb), dtype=torch.complex128, device=device)

for orb in range(n_orb):
    epsilon = -1.5 + orb * 1.0  # [-1.5, -0.5, 0.5, 1.5]
    for i, iwn in enumerate(omega):
        G_multi[i, orb, orb] = 1.0 / (iwn - epsilon)

G_base_multi = BaseTensor(tensor=G_multi, labels=['iwn', 'orb_i', 'orb_j'])

omega_multi = torch.linspace(-3, 3, 100, device=device)
omega_out, A_pade_multi = spectral.from_matsubara(
    G_base_multi, omega_multi, eta=0.05, method="pade",
    beta=beta, n_min=0, n_max=32
)

# Check each orbital
l2_errors_3 = []
for orb in range(n_orb):
    epsilon = -1.5 + orb * 1.0
    A_analytic_orb = torch.zeros_like(omega_multi)
    for i, w in enumerate(omega_multi):
        A_analytic_orb[i] = (0.05 / math.pi) / ((w - epsilon)**2 + 0.05**2)
    l2_errors_3.append(torch.sqrt(torch.mean((A_pade_multi[:, orb] - A_analytic_orb)**2)).item())

print(f"  L2 Errors by orbital:")
for orb, err in enumerate(l2_errors_3):
    print(f"    Orbital {orb}: {err:.10f}")
avg_l2_3 = sum(l2_errors_3) / len(l2_errors_3)
print(f"  Average L2 Error: {avg_l2_3:.10f}")
print(f"  Result: {'✓ PASS' if avg_l2_3 < 0.01 else '✗ FAIL'}")

# ========================================================================
# Test 4: DMFT-like Green's function with self-energy
# ========================================================================
print("\n" + "="*70)
print("TEST 4: DMFT-like (with Self-Energy)")
print("="*70)

# Simulate G(iω) = 1/(iω - ε - Σ(iω)) with Σ(iω) = U²/(iω)
U = 4.0
epsilon_f = 0.5  # f-orbital energy

G_dmft = torch.zeros((len(omega), 1, 1), dtype=torch.complex128, device=device)

for i, iwn in enumerate(omega):
    # IPT-like self-energy
    if i != len(omega) // 2:
        sigma = U**2 / iwn
    else:
        sigma = 0.0

    denom = iwn - epsilon_f - sigma
    if abs(denom) > 1e-12:
        G_dmft[i, 0, 0] = 1.0 / denom
    else:
        G_dmft[i, 0, 0] = 0.0

G_base_dmft = BaseTensor(tensor=G_dmft, labels=['iwn', 'orb_i', 'orb_j'])

# Use wider range for DMFT (Hubbard bands at ±U/2)
omega_dmft = torch.linspace(-10, 10, 200, device=device)
omega_out, A_pade_dmft = spectral.from_matsubara(
    G_base_dmft, omega_dmft, eta=0.05, method="pade",
    beta=beta, n_min=0, n_max=32
)

# For DMFT with atomic limit, expect Hubbard bands at ±U/2 = ±2
# Plus renormalized quasiparticle band
print(f"  DOS mean: {A_pade_dmft.mean():.6f}")
print(f"  DOS max: {A_pade_dmft.max():.6f}")
print(f"  Peak location: ω = {omega_dmft[A_pade_dmft.argmax()].item():.2f}")

# Check if peaks are in reasonable locations
peak_idx = A_pade_dmft.argmax()
peak_omega = omega_dmft[peak_idx].item()
has_peak = A_pade_dmft.max() > 1.0
peak_reasonable = abs(peak_omega) < 6.0  # Within expected range

print(f"  Has significant peak: {has_peak}")
print(f"  Peak in reasonable range: {peak_reasonable}")
print(f"  Result: {'✓ PASS' if has_peak and peak_reasonable else '✗ FAIL'}")

# ========================================================================
# Summary
# ========================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Test 1 (Simple Pole):     {'✓ PASS' if l2_error_1 < 0.01 else '✗ FAIL'} (L2 = {l2_error_1:.2e})")
print(f"  Test 2 (Bethe Lattice):   {'✓ PASS' if l2_error_2 < 0.1 else '✗ FAIL'} (L2 = {l2_error_2:.2e})")
print(f"  Test 3 (Multi-Orbital):   {'✓ PASS' if avg_l2_3 < 0.01 else '✗ FAIL'} (L2 = {avg_l2_3:.2e})")
print(f"  Test 4 (DMFT-like):       {'✓ PASS' if has_peak and peak_reasonable else '✗ FAIL'}")

all_pass = (
    l2_error_1 < 0.01 and
    l2_error_2 < 0.1 and
    avg_l2_3 < 0.01 and
    has_peak and peak_reasonable
)

print(f"\n  Overall: {'✓✓✓ ALL TESTS PASSED ✓✓✓' if all_pass else '✗✗✗ SOME TESTS FAILED ✗✗✗'}")
print("="*70)

# ========================================================================
# Plot
# ========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Test 1
axes[0, 0].plot(omega_real.cpu(), A_analytic.cpu(), 'k--', linewidth=2, label='Analytic')
axes[0, 0].plot(omega_real.cpu(), A_pade[:, 0].cpu(), 'r-', linewidth=2, label='Pade')
axes[0, 0].set_title(f'Test 1: Simple Pole\nL2 Error = {l2_error_1:.6f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Test 2
axes[0, 1].plot(omega_bethe.cpu(), A_bethe_broadened.cpu(), 'k--', linewidth=2, label='Analytic')
axes[0, 1].plot(omega_bethe.cpu(), A_pade_bethe[:, 0].cpu(), 'r-', linewidth=2, label='Pade')
axes[0, 1].set_title(f'Test 2: Bethe Lattice\nL2 Error = {l2_error_2:.6f}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Test 3
colors = ['blue', 'red', 'green', 'purple']
for orb in range(n_orb):
    epsilon = -1.5 + orb * 1.0
    A_analytic_orb = torch.zeros_like(omega_multi)
    for i, w in enumerate(omega_multi):
        A_analytic_orb[i] = (0.05 / math.pi) / ((w - epsilon)**2 + 0.05**2)
    axes[1, 0].plot(omega_multi.cpu(), A_analytic_orb.cpu(), '--', color=colors[orb], alpha=0.5)
    axes[1, 0].plot(omega_multi.cpu(), A_pade_multi[:, orb].cpu(), '-', color=colors[orb], label=f'Orb {orb}')
axes[1, 0].set_title(f'Test 3: Multi-Orbital\nAvg L2 Error = {avg_l2_3:.6f}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Test 4
axes[1, 1].plot(omega_dmft.cpu(), A_pade_dmft[:, 0].cpu(), 'g-', linewidth=2)
axes[1, 1].axvline(-U/2, color='k', linestyle='--', alpha=0.3, label=f'±U/2 = ±{U/2:.1f}')
axes[1, 1].axvline(U/2, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_title(f'Test 4: DMFT-like (U={U})\nPeak at ω = {peak_omega:.2f}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kagome_dmft_pade_test_comprehensive.png', dpi=100)
print(f"\nSaved plot to: kagome_dmft_pade_test_comprehensive.png")

sys.exit(0 if all_pass else 1)
