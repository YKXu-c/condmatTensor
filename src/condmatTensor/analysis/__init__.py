"""Analysis module: DOS, band structure, topology."""

from condmatTensor.analysis.bandstr import BandStructure
from condmatTensor.analysis.dos import DOSCalculator, ProjectedDOS
from condmatTensor.analysis.plotting_style import (
    DEFAULT_FIGURE_SIZES,
    DEFAULT_COLORS,
    DEFAULT_FONTSIZES,
    DEFAULT_STYLING,
    DEFAULT_COLORMAPS,
    LINE_STYLES,
    MARKER_STYLES,
)

__all__ = [
    "BandStructure",
    "DOSCalculator",
    "ProjectedDOS",
    # Plotting style constants
    "DEFAULT_FIGURE_SIZES",
    "DEFAULT_COLORS",
    "DEFAULT_FONTSIZES",
    "DEFAULT_STYLING",
    "DEFAULT_COLORMAPS",
    "LINE_STYLES",
    "MARKER_STYLES",
]
