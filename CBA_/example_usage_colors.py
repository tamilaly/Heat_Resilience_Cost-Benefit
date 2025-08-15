"""
Example usage of centralized color configuration
This file demonstrates different ways to import and use colors from config/colors.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Method 1: Import specific colors
from config.colors import COLOR_MAIN, COLOR_RED, COLOR_GREEN

# Method 2: Import all colors
from config.colors import *

# Method 3: Import the colors dictionary
from config.colors import COLORS

# Method 4: Import the color palette
from config.colors import COLOR_PALETTE

def example_plot_method1():
    """Example using specific imported colors"""
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, color=COLOR_MAIN, label='Sine', linewidth=2)
    plt.plot(x, y2, color=COLOR_RED, label='Cosine', linewidth=2)
    plt.fill_between(x, y1, y2, color=COLOR_GREEN, alpha=0.3)
    plt.title('Example Plot Using Specific Colors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def example_plot_method2():
    """Example using the colors dictionary"""
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, color=COLORS['purple'], label='Sine', linewidth=2)
    plt.plot(x, y2, color=COLORS['red'], label='Cosine', linewidth=2)
    plt.fill_between(x, y1, y2, color=COLORS['green'], alpha=0.3)
    plt.title('Example Plot Using Colors Dictionary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def example_plot_method3():
    """Example using the color palette for multiple lines"""
    x = np.linspace(0, 10, 100)
    
    plt.figure(figsize=(10, 6))
    for i in range(5):
        y = np.sin(x + i * 0.5)
        plt.plot(x, y, color=COLOR_PALETTE[i], 
                label=f'Line {i+1}', linewidth=2)
    
    plt.title('Example Plot Using Color Palette')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def example_bar_plot():
    """Example bar plot using centralized colors"""
    categories = ['Category A', 'Category B', 'Category C', 'Category D']
    values = [23, 45, 56, 78]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=COLOR_PALETTE[:len(categories)])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(value), ha='center', va='bottom')
    
    plt.title('Example Bar Plot Using Centralized Colors')
    plt.ylabel('Values')
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    print("Running color usage examples...")
    
    # Run examples
    example_plot_method1()
    example_plot_method2()
    example_plot_method3()
    example_bar_plot()
    
    print("Examples completed!") 