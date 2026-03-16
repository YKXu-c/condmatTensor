# Development Log - 2026-03-16

## Summary

Comprehensive code audit + bug fixes across core, manybody, and documentation files.

## Bugs Fixed

- [x] **HIGH** `analytic_continuation.py:461`: `lattice.dimension` → `lattice.dim` (AttributeError at runtime in `BetheLatticeContinuation._estimate_coordination_number`)
- [x] **MEDIUM** `impSolvers/ipt.py:_fft_to_iwn`: Removed spurious `/ self.beta` from forward Matsubara transform. Standard convention is `G(iωₙ) = ∫₀^β dτ e^{iωₙτ} G(τ)` (no 1/β prefactor); the 1/β belongs only in the inverse transform `G(τ) = (1/β) Σₙ ...`
- [x] **MEDIUM** `core/types.py:OrbitalMetadata.from_string()`: U-value parsing changed from `part_lower.startswith('u')` to `len(part) > 1 and part_lower[0] == 'u' and part_lower[1].isdigit()` — prevents collision with site names starting with 'U' (e.g., 'UCoGe')
- [x] **LOW** `core/types.py:OrbitalMetadata.is_f_orbital()`: Changed `'f' in self.orb.lower()` to `self.orb.lower().startswith('f')` — more precise matching
- [x] **DOC** `.cursorrules:174,179`: `TightBindingModel` → `HoppingModel` (class was renamed)

## Documentation Updated

- `CLAUDE.md`: Fixed API documentation (BareGreensFunction.compute, SelfEnergy.initialize_zero), added DMFT workflow, new imports, Known Bugs section, updated status to ~50%/7,500 lines
- `plans/architecture_plan.md`: Added "Known Bugs and Discrepancies" section with 10 items
- `plans/DEPENDENCY_ANALYSIS.md`: Added Section 8 with bug tables, updated file structure

## API Changes

- None (all changes are bug fixes, no API surface changed)

## Validation

- The IPT FFT normalization fix (removing `/ self.beta`) changes the magnitude of Σ(iωₙ). Previously the forward transform was scaled by 1/β² effectively (dtau/β = 1/n_tau / β × β = 1/n_tau was correct but then divided by β again). After fix, Σ magnitudes will be β× larger than before. Downstream DMFT convergence should be re-validated by running `examples/kagome_f_dmft.py`.
- The `is_f_orbital` fix is backward-compatible for all standard orbital names ('f', 'f0'...'f6').

## Next Steps

1. Re-run `examples/kagome_f_dmft.py` to verify DMFT convergence after IPT normalization fix
2. Run `examples/pade_simple_test.py` to check Pade analytic continuation still works
3. Consider adding a round-trip test for IPT FFT: G(iωₙ) → G(τ) → G(iωₙ) should recover original
