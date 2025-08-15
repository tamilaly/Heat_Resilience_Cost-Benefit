import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde
import scipy.stats as stats
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator

# Import colors from the centralized color configuration
from config.colors import (
    COLOR_MAIN, COLOR_HIST, COLOR_PURPLE, COLOR_BLUE, COLOR_RED, 
    COLOR_GREEN, COLOR_PINK, COLOR_0, COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5,
    HIGHLIGHT_1, HIGHLIGHT_2, HIGHLIGHT_3
)


# load directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# results folder from date
folder = "25-08-15"

file_csv = os.path.join(current_dir, "results", folder, "TCOO_statistics.csv")

# load csv
df = pd.read_csv(file_csv)

# read only rows where cooling==True and heating==False
df_cooling = df[df["cooling_enabled"] == True]
df_only_cooling = df_cooling[df_cooling["heating_enabled"] == False]

# load the values for the mean , threshold_8_years, 99th_percentile
mean_cooling = df_only_cooling["mean_tcoo"].values
threshold_8_years_cooling = df_only_cooling["threshold_8_years"].values
ninety_ninth_percentile_cooling = df_only_cooling["99th_percentile"].values

# compare the distance to the mean for each scenario
distance_threshold_8_years = threshold_8_years_cooling - mean_cooling

distance_99th_percentile = ninety_ninth_percentile_cooling - mean_cooling

# make x the scenario name
x = df_only_cooling["scenario_name"].values
x_axis = np.arange(len(x))

# remove scenario name -extra
x = [i.replace("-extra", "") for i in x]

for i in range(len(x_axis)):
    print(f"{x[i]} - distance to threshold 8 years: {round(distance_threshold_8_years[i], 2)} €")
    print(f"{x[i]} - distance to 99th percentile: {round(distance_99th_percentile[i], 2)} €")

# plot the distance to the mean for each scenario
plt.figure(figsize=(10, 6))

# Create offset x positions
x_offset = 0.1
x_threshold = x_axis - x_offset
x_percentile = x_axis + x_offset

markerline, stemlines, baseline = plt.stem(x_threshold, distance_threshold_8_years)
plt.setp(markerline, marker='o', color=HIGHLIGHT_2, label="Dispersion for exceptional extremes (8 years)")
plt.setp(stemlines, color=HIGHLIGHT_2)    # Stem lines color
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
baseline.set_visible(False)  # Hide the baseline

markerline, stemlines, baseline = plt.stem(x_percentile, distance_99th_percentile)
plt.setp(markerline, marker='o', color=HIGHLIGHT_1, label="Dispersion for 99th percentile")
plt.setp(stemlines, color=HIGHLIGHT_1)    # Stem lines color
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
baseline.set_visible(False)  # Hide the baseline

plt.axhline(y=distance_threshold_8_years[0], color='lightgray', linestyle='--', linewidth=0.7, zorder=0)
plt.axhline(y=distance_99th_percentile[0], color='lightgray', linestyle='--', linewidth=0.7, zorder=0)
plt.grid(True, color='lightgray', axis='y')
plt.xticks(x_axis, x, rotation=45, ha="right")
plt.ylabel("Cost Difference (€)")
plt.title("Additional Expected Cost - Dispersion from Mean for Each Scenario")
plt.ylim(0, 4500)
plt.legend()
plt.show()
