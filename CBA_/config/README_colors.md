# Centralized Color Configuration

This directory contains a centralized color configuration system for the CBA project to ensure consistent colors across all Python files.

## Files

- `colors.py` - Main color definitions file
- `README_colors.md` - This documentation file

## Available Colors

### Main Colors
- `COLOR_PURPLE` - (112/255, 44/255, 168/255) - Main purple
- `COLOR_BLUE` - (5/255, 0/255, 113/255) - Dark blue
- `COLOR_RED` - (204/255, 31/255, 4/255) - Red
- `COLOR_GREEN` - (98/255, 177/255, 163/255) - Green
- `COLOR_PINK` - (237/255, 145/255, 185/255) - Pink

### Derived Colors
- `COLOR_MAIN` - Same as COLOR_PURPLE
- `COLOR_HIST` - Lighter version of COLOR_MAIN for histograms

### Sequential Colors (for multiple plots)
- `COLOR_0` - (242/255, 99/255, 102/255) - Light red
- `COLOR_1` - (243/255, 182/255, 68/255) - Yellow
- `COLOR_2` - 'darkorange' - Orange
- `COLOR_3` - (98/255, 177/255, 163/255) - Teal green
- `COLOR_4` - (102/255, 153/255, 255/255) - Light blue
- `COLOR_5` - (102/255, 255/255, 102/255) - Light green

### Additional Colors
- `COLOR_LIGHT_GRAY` - (0.8, 0.8, 0.8)
- `COLOR_DARK_GRAY` - (0.3, 0.3, 0.3)
- `COLOR_WHITE` - (1.0, 1.0, 1.0)
- `COLOR_BLACK` - (0.0, 0.0, 0.0)

### Collections
- `COLOR_PALETTE` - List of [COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5]
- `COLORS` - Dictionary with named access to all colors

## Usage Examples

### Method 1: Import Specific Colors
```python
from config.colors import COLOR_MAIN, COLOR_RED, COLOR_GREEN

plt.plot(x, y, color=COLOR_MAIN)
plt.fill_between(x, y1, y2, color=COLOR_GREEN, alpha=0.3)
```

### Method 2: Import All Colors
```python
from config.colors import *

plt.plot(x, y, color=COLOR_PURPLE)
plt.scatter(x, y, color=COLOR_BLUE)
```

### Method 3: Use Colors Dictionary
```python
from config.colors import COLORS

plt.plot(x, y, color=COLORS['purple'])
plt.bar(categories, values, color=COLORS['green'])
```

### Method 4: Use Color Palette for Multiple Plots
```python
from config.colors import COLOR_PALETTE

for i, data in enumerate(datasets):
    plt.plot(x, data, color=COLOR_PALETTE[i], label=f'Dataset {i+1}')
```

## Benefits

1. **Consistency** - All plots use the same color scheme
2. **Maintainability** - Change colors in one place
3. **Professional Look** - Consistent branding across all visualizations
4. **Easy Updates** - Modify colors without searching through multiple files

## Adding New Colors

To add new colors:

1. Add the color definition to `colors.py`
2. Add it to the `COLORS` dictionary if you want named access
3. Update this README if needed

Example:
```python
# In colors.py
COLOR_NEW = (100/255, 150/255, 200/255)

# Add to COLORS dictionary
COLORS = {
    # ... existing colors ...
    'new_color': COLOR_NEW
}
```

## Migration Guide

To migrate existing files:

1. Remove local color definitions
2. Add import statement: `from config.colors import COLOR_MAIN, COLOR_RED, ...`
3. Replace color references with imported colors
4. Test to ensure plots look the same

## Notes

- All RGB values are normalized to 0-1 range for matplotlib compatibility
- The `COLOR_HIST` is automatically calculated as a lighter version of `COLOR_MAIN`
- Use `COLOR_PALETTE` for plots with multiple lines/series
- The `COLORS` dictionary provides easy access by name 