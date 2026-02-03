"""Verify the Pade continued fraction evaluation formula.

The standard Vidberg-Serene continued fraction is:
C(z) = a₀ / (1 + a₁(z-z₀) / (1 + a₂(z-z₁) / (1 + a₃(z-z₂) / (...))))

This can be evaluated bottom-up as:
result = a[N-1]
for i = N-2 down to 0:
    result = a[i] * (z - z[i]) / (1 + result)

BUT: the current code uses (z - z[i-1]) instead of (z - z[i])!

Let's verify which is correct.
"""

import torch
import math
import sys
sys.path.insert(0, 'src')

device = torch.device('cpu')

# Simple test case: G(z) = 1/(z + 1) has known continued fraction
# For this simple case, we can test the evaluation formula directly

# Use a simple rational function: R(z) = 1/(1+z)
# We'll construct a Pade approximant from this

# First, generate some G values at Matsubara frequencies
beta = 10.0
N = 5  # Small number for easy manual verification

n_vals = torch.arange(N, device=device)
wn = (2 * n_vals + 1) * math.pi / beta
z_iwn = 1j * wn

print("="*60)
print("Verifying Pade Evaluation Formula")
print("="*60)

print(f"\nMatsubara frequencies:")
for i in range(N):
    print(f"  i={i}, n={n_vals[i]}, z={z_iwn[i]}")

# Generate G(z) = 1/(z + 1) at these frequencies
G_vals = torch.zeros(N, dtype=torch.complex128, device=device)
for i in range(N):
    G_vals[i] = 1.0 / (z_iwn[i] + 1.0)

print(f"\nG values:")
for i in range(N):
    print(f"  i={i}, G={G_vals[i]}")

# Build the g-table (Vidberg-Serene)
g = torch.zeros((N, N), dtype=torch.complex128, device=device)
g[0, :] = G_vals

for i in range(1, N):
    for j in range(i, N):
        numerator = g[i-1, i-1] - g[i-1, j]
        denominator = (z_iwn[j] - z_iwn[i-1]) * g[i-1, j]
        if abs(denominator) > 1e-12:
            g[i, j] = numerator / denominator

print(f"\ng-table diagonal (coefficients a):")
for i in range(N):
    print(f"  a[{i}] = g[{i},{i}] = {g[i,i]}")

a = g.diag()

# Test at a real frequency: z = 0.5j
z_test = 0.5j

print(f"\n{'='*60}")
print(f"Testing at z = {z_test}")
print(f"{'='*60}")

# Method 1: Direct evaluation using the original function
G_exact = 1.0 / (z_test + 1.0)
print(f"\nExact value: G({z_test}) = {G_exact}")
print(f"  Im[G] = {G_exact.imag}")

# Method 2: Current code formula (using z - z[i-1])
result1 = 0j
for i in range(N-1, 0, -1):
    result1 = a[i] * (z_test - z_iwn[i-1]) / (1.0 + result1)
G_pade1 = a[0] / (1.0 + result1)
print(f"\nCurrent formula (z - z[i-1]):")
print(f"  G_pade = {G_pade1}")
print(f"  Im[G] = {G_pade1.imag}")
print(f"  Error = {abs(G_pade1 - G_exact)}")

# Method 3: Proposed fix (using z - z[i])
result2 = 0j
for i in range(N-1, 0, -1):
    result2 = a[i] * (z_test - z_iwn[i]) / (1.0 + result2)
G_pade2 = a[0] / (1.0 + result2)
print(f"\nProposed formula (z - z[i]):")
print(f"  G_pade = {G_pade2}")
print(f"  Im[G] = {G_pade2.imag}")
print(f"  Error = {abs(G_pade2 - G_exact)}")

# Method 4: Standard continued fraction formula
# C(z) = a₀ / (1 + a₁(z-z₀) / (1 + a₂(z-z₁) / (...)))
# Start from innermost: D[N-2] = 1 + a[N-1](z-z[N-2])
# Then: D[N-3] = 1 + a[N-2](z-z[N-3]) / D[N-2]
# etc.
result3 = 1.0 + a[N-1] * (z_test - z_iwn[N-2])
for i in range(N-2, 0, -1):
    result3 = 1.0 + a[i] * (z_test - z_iwn[i-1]) / result3
G_pade3 = a[0] / result3
print(f"\nStandard CF formula:")
print(f"  G_pade = {G_pade3}")
print(f"  Im[G] = {G_pade3.imag}")
print(f"  Error = {abs(G_pade3 - G_exact)}")

print(f"\n{'='*60}")
print(f"Summary:")
print(f"{'='*60}")
print(f"Exact Im[G]: {G_exact.imag:.10f}")
print(f"Current (z - z[i-1]): {G_pade1.imag:.10f} (error: {abs(G_pade1 - G_exact):.2e})")
print(f"Proposed (z - z[i]): {G_pade2.imag:.10f} (error: {abs(G_pade2 - G_exact):.2e})")
print(f"Standard CF: {G_pade3.imag:.10f} (error: {abs(G_pade3 - G_exact):.2e})")

# Test at multiple points
print(f"\n{'='*60}")
print(f"Testing at multiple points:")
print(f"{'='*60}")

test_points = [0.1j, 0.5j, 1.0j, 2.0j, -0.5j, -1.0j]
print(f"\n{'z':>10s} {'Exact':>12s} {'Current':>12s} {'Proposed':>12s} {'Standard':>12s}")
print(f"{'='*70}")

for z_test in test_points:
    G_exact = 1.0 / (z_test + 1.0)

    # Current
    result1 = 0j
    for i in range(N-1, 0, -1):
        result1 = a[i] * (z_test - z_iwn[i-1]) / (1.0 + result1)
    G_pade1 = a[0] / (1.0 + result1)

    # Proposed
    result2 = 0j
    for i in range(N-1, 0, -1):
        result2 = a[i] * (z_test - z_iwn[i]) / (1.0 + result2)
    G_pade2 = a[0] / (1.0 + result2)

    # Standard
    result3 = 1.0 + a[N-1] * (z_test - z_iwn[N-2])
    for i in range(N-2, 0, -1):
        result3 = 1.0 + a[i] * (z_test - z_iwn[i-1]) / result3
    G_pade3 = a[0] / result3

    print(f"{str(z_test):>10s} {G_exact.imag:>12.8f} {G_pade1.imag:>12.8f} {G_pade2.imag:>12.8f} {G_pade3.imag:>12.8f}")
