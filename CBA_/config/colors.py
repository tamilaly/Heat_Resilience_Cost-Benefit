# Color definitions for the CBA project
# RGB values normalized to 0-1 range

# Main colors
COLOR_PURPLE = (112/255, 44/255, 168/255)
COLOR_BLUE = (5/255, 0/255, 113/255)
COLOR_RED = (204/255, 31/255, 4/255)
COLOR_GREEN = (98/255, 177/255, 163/255)
COLOR_PINK = (237/255, 145/255, 185/255)

# Main color and histogram color
COLOR_MAIN = COLOR_PURPLE
COLOR_HIST = tuple(0.5 * (c + 1) for c in COLOR_MAIN)

# Highlight colors
HIGHLIGHT_1 = (204/255, 31/255, 4/255)
HIGHLIGHT_2 = (237/255, 145/255, 185/255)
HIGHLIGHT_3 = (277/255, 144/255, 132/255)

# Sequential colors for multiple plots
COLOR_0 = 242/255, 99/255, 102/255
COLOR_1 = (243/255, 182/255, 68/255)
COLOR_2 = 'darkorange'
COLOR_3 = (98/255, 177/255, 163/255)
COLOR_4 = (102/255, 153/255, 255/255)
COLOR_5 = (102/255, 255/255, 102/255)

# Color palette for multiple scenarios
COLOR_PALETTE = [COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5]

# Additional colors if needed
COLOR_LIGHT_GRAY = (0.8, 0.8, 0.8)
COLOR_DARK_GRAY = (0.3, 0.3, 0.3)
COLOR_WHITE = (1.0, 1.0, 1.0)
COLOR_BLACK = (0.0, 0.0, 0.0)

# Dictionary for easy access by name
COLORS = {
    'purple': COLOR_PURPLE,
    'blue': COLOR_BLUE,
    'red': COLOR_RED,
    'green': COLOR_GREEN,
    'pink': COLOR_PINK,
    'main': COLOR_MAIN,
    'hist': COLOR_HIST,
    'yellow': COLOR_1,
    'orange': COLOR_2,
    'light_gray': COLOR_LIGHT_GRAY,
    'dark_gray': COLOR_DARK_GRAY,
    'white': COLOR_WHITE,
    'black': COLOR_BLACK
} 