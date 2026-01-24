"""Standardized plotting style constants for condmatTensor.

This module provides publication-quality default styling for plots generated
by the condmatTensor library. All constants can be overridden via **kwargs
in plotting methods.

Example:
    >>> from condmatTensor.analysis.plotting_style import DEFAULT_COLORS
    >>> fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZES['single'])
    >>> ax.plot(x, y, color=DEFAULT_COLORS['primary'])
"""

# Figure sizes for different plot types
DEFAULT_FIGURE_SIZES = {
    'single': (6, 5),
    'dual': (14, 5),
    'triple': (15, 5),
    '2x2': (12, 10),
    'band_dos': (10, 5),
    'comparison_2': (14, 5),
    'comparison_3': (15, 5),
    'comparison_2x3': (15, 10),
}

# Color scheme for plots
DEFAULT_COLORS = {
    'primary': '#3498db',
    'reference': '#e74c3c',
    'secondary': '#2ecc71',
    'tertiary': '#f39c12',
    'quaternary': '#9b59b6',
    'cpu': '#3498db',
    'gpu': '#e74c3c',
    'flat_band': '#e74c3c',
    'fermi_level': '#27ae60',
    'fill_default': 'skyblue',
}

# Font sizes for different text elements
DEFAULT_FONTSIZES = {
    'labels': 12,
    'titles': 12,
    'legend': 10,
    'annotation': 9,
    'tick': 11,
    'suptitle': 14,
}

# Styling options for plot elements
DEFAULT_STYLING = {
    'grid_alpha': 0.3,
    'band_linewidth': 1.0,
    'band_alpha': 1.0,
    'overlay_alpha': 0.6,
    'dpi': 150,
    'fill_alpha': 0.3,
    'reference_linewidth': 1.5,
    'scatter_size': 10,
}

# Default colormaps for different data types
DEFAULT_COLORMAPS = {
    'orbital_weight': 'viridis',
    'spin': 'RdBu_r',
    'orbitals': 'tab10',
    'sequential': 'viridis',
    'diverging': 'RdBu_r',
    'categorical': 'tab10',
}

# Line styles for reference lines and annotations
LINE_STYLES = {
    'solid': '-',
    'dashed': '--',
    'dotted': ':',
    'dash_dot': '-.',
}

# Marker styles for scatter plots
MARKER_STYLES = {
    'circle': 'o',
    'square': 's',
    'triangle': '^',
    'diamond': 'D',
    'cross': 'x',
    'plus': '+',
}
