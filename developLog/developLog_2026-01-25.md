# Development Log - 2026-01-25

## Rename: TightBindingModel → HoppingModel

### Summary
Renamed `TightBindingModel` class to `HoppingModel` because the class does not enforce "tight-binding" constraints (nearest-neighbor only). The new name more accurately describes the class as a general hopping model builder that supports arbitrary hoppings at any distance.

### Rationale
The term "tight-binding" typically refers to models with nearest-neighbor hopping only. However, the `TightBindingModel` class supports arbitrary hoppings at any distance, making the name misleading. The new name `HoppingModel` better reflects the class's general functionality.

### Files Modified

#### Core Source Code (2 files)
- `src/condmatTensor/lattice/model.py`
  - Line 7: Renamed class `TightBindingModel` → `HoppingModel`
  - Line 33: Updated docstring "Initialize TightBindingModel" → "Initialize HoppingModel"
  - Line 228: Updated return type annotation `-> TightBindingModel` → `-> HoppingModel`
  - Updated all docstrings and examples

- `src/condmatTensor/lattice/__init__.py`
  - Line 3: Updated import: `TightBindingModel` → `HoppingModel`
  - Line 6: Updated `__all__` export: `"TightBindingModel"` → `"HoppingModel"`

#### Example Files (12 files)
All example files updated with:
- Import statements: `TightBindingModel` → `HoppingModel`
- Type annotations: `-> TightBindingModel` → `-> HoppingModel`
- Instantiations: `TightBindingModel(...)` → `HoppingModel(...)`
- Docstrings and comments

Files:
1. `examples/kagome_bandstructure.py`
2. `examples/kagome_with_f_bandstructure.py`
3. `examples/kagome_spinful_bandstructure.py`
4. `examples/kagome_f_effective_array.py`
5. `examples/gpu_performance_benchmark.py`
6. `examples/bayesian_parameter_sweep_example.py`
7. `examples/kagome_spinful_with_b_field.py`
8. `examples/kagome_f_spinful_bandstructure.py`
9. `examples/kagome_f_spinful_effective.py`
10. `examples/test_magnetic_additional.py`
11. `examples/orbital_metadata_demo.py`

#### Documentation Files (7 files)
- `CLAUDE.md` - Updated import examples and status table
- `.cursorrules` - Updated architecture overview and import patterns
- `.github/copilot-instructions.md` - Updated architecture and examples
- `developLog/developLog_2026-01-24.md` - Updated status table
- `developLog/allAPI.md` - Updated all API documentation references
- `plans/architecture_plan.md` - Updated all references throughout
- `plans/DEPENDENCY_ANALYSIS.md` - Updated class references

### Verification
- ✅ Import statement works: `from condmatTensor.lattice import HoppingModel`
- ✅ Functionality test: HoppingModel can create models and add hoppings
- ✅ All example files updated
- ✅ Documentation files updated

### Notes
- This is a pure refactoring - no functional changes
- The rename affects the public API, so all dependent code must be updated
- No backward compatibility alias is needed since this is an active development branch
