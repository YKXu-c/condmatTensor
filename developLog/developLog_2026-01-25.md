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

---

## Plotting Upgrade: analysis.dos and analysis.bandstr

### Summary

Implemented publication-quality plotting enhancements for LEVEL 5 (Analysis) module, adding new plotting methods to `BandStructure` and `DOSCalculator` classes, along with a standardized plotting style constants module.

### New Features

#### 1. Plotting Style Constants Module

**File**: `src/condmatTensor/analysis/plotting_style.py` (NEW, 70 lines)

Standardized constants for publication-quality plotting:
- `DEFAULT_FIGURE_SIZES`: Predefined figure sizes for common plot types
- `DEFAULT_COLORS`: Color scheme for primary, reference, secondary, etc.
- `DEFAULT_FONTSIZES`: Font sizes for labels, titles, legend, etc.
- `DEFAULT_STYLING`: Grid alpha, linewidth, DPI settings, etc.
- `DEFAULT_COLORMAPS`: Colormaps for orbital weight, spin, etc.
- `LINE_STYLES`: Line styles (solid, dashed, dotted, dash-dot)
- `MARKER_STYLES`: Marker styles for scatter plots

#### 2. BandStructure Class Enhancements

**File**: `src/condmatTensor/analysis/bandstr.py` (~550 lines, added ~310 lines)

New methods:
- `plot_colored_by_weight()`: Color bands by orbital weight using scatter plot
- `add_reference_line()`: Add horizontal reference line
- `plot_comparison()`: Overlay multiple band structures
- `plot_multi_panel()`: Create multi-panel band structure comparison

#### 3. DOSCalculator Class Enhancements

**File**: `src/condmatTensor/analysis/dos.py` (~520 lines, added ~210 lines)

New methods:
- `plot_with_reference()`: Plot DOS with vertical reference lines
- `plot_comparison()`: Overlay multiple DOS curves
- `plot_multi_panel()`: Create multi-panel DOS comparison

#### 4. Module Exports Update

**File**: `src/condmatTensor/analysis/__init__.py` (updated)

Added exports for plotting style constants.

### Files Modified

| File | Lines | Change |
|------|-------|--------|
| `src/condmatTensor/analysis/plotting_style.py` | 70 | NEW |
| `src/condmatTensor/analysis/bandstr.py` | 550 | +310 lines (new methods) |
| `src/condmatTensor/analysis/dos.py` | 520 | +210 lines (new methods) |
| `src/condmatTensor/analysis/__init__.py` | 22 | +15 lines (exports) |
| `developLog/allAPI.md` | 1310 | +130 lines (documentation) |

### Design Decisions

1. **Return axes, not figures**: Consistent with current API, allows user customization
2. **Defaults via constants**: Provide sensible defaults via `plotting_style.py`, allow override via **kwargs
3. **Colormap defaults**: 'viridis' for orbital weight (perceptually uniform), 'tab10' for categorical
4. **Backward compatibility**: Did not modify existing `plot()` and `plot_with_dos()` methods
5. **CPU/GPU split**: Used `.cpu().numpy()` before matplotlib calls for GPU tensors

### Next Steps

Phase 3 of the plan involves refactoring kagome examples to use new plotting methods:
1. `kagome_bandstructure.py` - Use `add_reference_line()`, `plot_multi_panel()`
2. `kagome_f_spinful_bandstructure.py` - Use `plot_colored_by_weight()`
3. `kagome_f_effective_array.py` - Use `plot_comparison()`
4. Other examples - Apply new one-line plotting methods
