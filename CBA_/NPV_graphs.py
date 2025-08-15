import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import scipy.stats as stats

# Import colors from the centralized color configuration
from config.colors import (
    COLOR_MAIN, COLOR_HIST, COLOR_PURPLE, COLOR_BLUE, COLOR_RED, 
    COLOR_GREEN, COLOR_PINK, COLOR_0, COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5
)

def find_percentiles(kde, x_vals, values):
    
    # KDE estimation
    pdf_kde = kde(x_vals)
    cdf_kde = np.cumsum(pdf_kde)
    cdf_kde = cdf_kde / cdf_kde[-1]
    
    # Interpolation function: CDF(x) → x
    cdf = interp1d(x_vals, cdf_kde)

    percentile_values = []
    for v in values:
        # Get x-values for given percentiles
        percentile_values.append(float(cdf(v)))

    return percentile_values


def plot_single_npv_distribution(file_path, scenario_name, threshold_1=None, limit=8, marker=False, output_path=None, bw_method=0.6):
    """
    Plot NPV distribution from a single file and optionally save as PNG.
    
    Args:
        file_path (str): Path to the CSV file containing NPV data
        scenario_name (str): Name of the scenario for the plot title
        output_path (str, optional): Path to save the PNG file. If None, plot is only displayed
        bw_method (float): Bandwidth method for KDE
    """
    df = pd.read_csv(file_path)
    
    if marker == True:
            # Parse the energy cost string into a list of numbers for each row
        df['Energy_Savings_List'] = df['Cooling_Savings'].apply(lambda x: [float(i) for i in x.strip('"').split(',')])

        # Count energy costs above threshold for each row
        df['Energy_Savings_Above_Threshold'] = df['Energy_Savings_List'].apply(lambda x: sum(1 for i in x if i >= threshold_1))

        # Get TCOO values
        NPV_list = df['Final NPV'].values
        energy_savings_above_threshold = df['Energy_Savings_Above_Threshold'].values
        print(f"Mean number of savings above threshold: {np.mean(energy_savings_above_threshold)}")

        # find the NPV for the limit
        NPV_values_at_limit_1 = NPV_list[energy_savings_above_threshold == limit]
        print(f"Minimum NPV values with {limit} years: {np.mean(NPV_values_at_limit_1):,.0f} Euro")
        limit_up = np.mean(NPV_values_at_limit_1)
    else:
        # Parse the energy cost string into a list of numbers for each row
        df['Energy_Savings_List'] = df['Cooling_Savings'].apply(lambda x: [float(i) for i in x.strip('"').split(',')])

        # Count energy costs above threshold for each row
        df['Energy_Savings_Above_Threshold'] = df['Energy_Savings_List'].apply(lambda x: sum(1 for i in x if i >= threshold_1))
        # Get TCOO values
        NPV_list = df['Final NPV'].values
        energy_savings_above_threshold = df['Energy_Savings_Above_Threshold'].values
        NPV_values_at_limit = NPV_list[energy_savings_above_threshold >= limit]
        
        # Add debugging information
        print(f"Number of NPV values at limit {limit}: {len(NPV_values_at_limit)}")
        if len(NPV_values_at_limit) > 0:
            print(f"NPV values at limit")
        else:
            print("No NPV values found at the specified limit")

    # Gaussian KDE
    if bw_method == None:
        kde = gaussian_kde(df["Final NPV"])
    else:
        kde = gaussian_kde(df["Final NPV"], bw_method=bw_method)
    x_vals = np.linspace(min(df["Final NPV"]), max(df["Final NPV"]), 1000)
    y_vals = kde(x_vals)

    if marker == True:
        # find percentiles at limit_up and limit_down
        percentile_values = find_percentiles(kde, x_vals, [limit_up])
        prob_down = percentile_values[0]
        prob_up = 1 - percentile_values[0]
        print(f"Probability of {limit_up} NPV or more: {prob_up*100:.0f}%")

    # Calculate the mean and standard deviation of the NPV distribution
    mean_npv = df["Final NPV"].mean()
    std_npv = df["Final NPV"].std()

    # Central 95% interval of the NPV distribution
    lower_bound = np.percentile(df["Final NPV"], 2.5)
    upper_bound = np.percentile(df["Final NPV"], 97.5)
    first_percentile = np.percentile(df["Final NPV"], 1)
    ninety_ninth_percentile = np.percentile(df["Final NPV"], 99)

    # Calculate the proportion of negative values within the confidence interval
    negative_values = df["Final NPV"][(df["Final NPV"] >= lower_bound) & (df["Final NPV"] <= upper_bound) & (df["Final NPV"] < 0)]
    total_values = df["Final NPV"][(df["Final NPV"] >= lower_bound) & (df["Final NPV"] <= upper_bound)]
    overlap_proportion = len(negative_values) / len(total_values) if len(total_values) > 0 else 0

    # Calculate total proportion of negative values
    total_negative = df["Final NPV"][df["Final NPV"] < 0]
    total_overlap = len(total_negative) / len(df["Final NPV"])

    # Print the mean and confidence intervals
    print(f"Mean NPV: {mean_npv:.2f} Eur")
    print(f"Standard Deviation: {std_npv:.2f} Eur")
    print(f"99th Percentile: {ninety_ninth_percentile:.2f} Eur")
    print(f"95% CI: {lower_bound:.2f} to {upper_bound:.2f} Eur")
    print(f"Overlap with negative values within CI: {overlap_proportion:.2%}")
    print(f"Total overlap with negative values: {total_overlap:.2%}")

    if marker == True:
        print(f"Distance to 15 years threshold: {limit_up - mean_npv:.2f} Eur")
        print(f"Distance to 99th percentile: {ninety_ninth_percentile - mean_npv:.2f} Eur")

    # Plot the NPV distribution
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, y_vals, color=COLOR_MAIN, label="KDE")
    plt.hist(df["Final NPV"], bins=100, color=COLOR_HIST, alpha=0.5, label="Histogram", density=True)
    

    if marker == True:
        # Add vertical line for mean
        plt.axvline(x=mean_npv, color=COLOR_GREEN, linestyle='--', alpha=0.7, label=f'NPV Mean = {mean_npv:,.0f} €')
        
        # fill the middle with hist color
        mask_3 = (x_vals <= limit_up)
        plt.fill_between(x_vals[mask_3], 0, y_vals[mask_3], color=COLOR_GREEN, alpha=0.3, label=f'low to average outcomes')
        prob_mid = prob_down
        place= limit_up - (limit_up- lower_bound)/2
        plt.text(place, 0.0007, f'{prob_mid*100:.0f}%', color=COLOR_GREEN, fontsize=12)

        # plot the marker with axv line
        plt.axvline(x=limit_up, color=COLOR_PINK, linestyle='--', alpha=0.7, label=f'{limit} years threshold = {limit_up:,.0f} €')
        # shade the area above the curve from marker to max
        mask = x_vals >= limit_up
        plt.fill_between(x_vals[mask], 0, y_vals[mask], color=COLOR_PINK, alpha=0.3, label=f'>= {limit} years with more than {threshold_1} € savings')
        place= limit_up + (upper_bound- limit_up)/2
        plt.text(place, 0.0007, f'{prob_up*100:.0f}%', color=COLOR_PINK, fontsize=12)

        plt.axvline(x=ninety_ninth_percentile, color=COLOR_RED, linestyle='--', alpha=0.7, label=f'99th Percentile = {ninety_ninth_percentile:,.0f} €')
    else:
        plt.axvline(x=mean_npv, color=COLOR_MAIN, linestyle='--',label=f'NPV Mean = {mean_npv:,.0f} €')
        # Add vertical lines for confidence interval
        # plt.axvline(x=lower_bound, color=COLOR_MAIN, linestyle=':', alpha=0.7, label='95% Confidence Interval')
        # plt.axvline(x=upper_bound, color=COLOR_MAIN, linestyle=':', alpha=0.7)

        # plt.axvline(x=first_percentile, color=COLOR_RED, linestyle='--', alpha=0.7, label=f'1st Percentile = {first_percentile:,.0f} €')
        # plt.axvline(x=ninety_ninth_percentile, color=COLOR_RED, linestyle='--', alpha=0.7, label=f'99th Percentile = {ninety_ninth_percentile:,.0f} €')

        # Plot vertical lines for NPV values at limit
        if len(NPV_values_at_limit) > 0:
            for i in range(len(NPV_values_at_limit)):
                plt.axvline(x=NPV_values_at_limit[i], color=COLOR_PINK, linestyle=':', linewidth=0.8, 
                           label='NPV with >=15 years above threshold' if i==0 else '')

    
    # scenario names remove _ & -extra in case of -extra
    scenario_name = scenario_name.replace("_", " ")
    scenario_name = scenario_name.replace("-extra", "")

    plt.title(f"Cooling NPV Distribution for {scenario_name}")
    plt.xlabel("NPV")
    plt.ylabel("Frequency")
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# plots for comparison of scenarios
def plot_npv_boxplots(file_paths, scenario_names, output_path=None, figsize=(16, 8)):
    """
    Create boxplots comparing NPVs across different scenarios.
    
    Args:
        file_paths (list): List of paths to CSV files containing NPV data for each scenario
        scenario_names (list): List of scenario names corresponding to each file
        output_path (str, optional): Path to save the PNG file. If None, plot is only displayed
        figsize (tuple): Figure size as (width, height) in inches
    """
    # Read all data files
    data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data.append(df["Final NPV"])
    
    # remove -extra in case of -extra
    scenario_names = [name.replace("-extra", "") for name in scenario_names]

    # Create the boxplot
    plt.figure(figsize=figsize)
    boxplot = plt.boxplot(data, labels=scenario_names, patch_artist=True, showfliers=False)
    # Customize boxplot colors
    colors = [COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5]
    for patch, color in zip(boxplot['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
    
    # Set median line color to black
    for median in boxplot['medians']:
        median.set_color('black')
    
    # Add labels and title
    plt.ylabel('NPV')
    plt.title('NPV Distribution Comparison for Heating-Cooling')
    
    # Rotate x-axis labels if they're long
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
def plot_npv_distribution_bars(file_paths, scenario_names, output_path=None, figsize=(14, 6)):
    """
    Create distribution bars comparing NPVs across different scenarios in landscape format.
    
    Args:
        file_paths (list): List of paths to CSV files containing NPV data for each scenario
        scenario_names (list): List of scenario names corresponding to each file
        output_path (str, optional): Path to save the PNG file. If None, plot is only displayed
        figsize (tuple): Figure size as (width, height) in inches
    """
    # Read all data files
    data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data.append(df["Final NPV"])
    
    # remove -extra in case of -extra
    scenario_names = [name.replace("-extra", "") for name in scenario_names]

    # Calculate statistics for each scenario
    means = [np.mean(d) for d in data]
    medians = [np.median(d) for d in data]
    q25 = [np.percentile(d, 25) for d in data]
    q75 = [np.percentile(d, 75) for d in data]
    q10 = [np.percentile(d, 10) for d in data]
    q90 = [np.percentile(d, 90) for d in data]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors
    colors = [COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5]
    
    # Set up x positions
    x_pos = np.arange(len(scenario_names))
    bar_width = 0.6
    
    # Plot the main distribution bars (25th to 75th percentile)
    bars = ax.bar(x_pos, [q75[i] - q25[i] for i in range(len(data))], 
                  bottom=q25, width=bar_width, 
                  color=colors[:len(data)], alpha=0.8, 
                  label='25th-75th percentile')
    
    # Plot the extended distribution bars (10th to 90th percentile)
    ax.bar(x_pos, [q90[i] - q10[i] for i in range(len(data))], 
           bottom=q10, width=bar_width, 
           color=colors[:len(data)], alpha=0.3, 
           label='10th-90th percentile')
    
    # Add median lines
    ax.plot(x_pos, medians, 'k_', markersize=15, linewidth=3, 
            label='Median', zorder=5)
    
    # Add mean markers
    ax.plot(x_pos, means, 'ko', markersize=8, 
            label='Mean', zorder=6)
    
    # Customize the plot
    # ax.set_xlabel('Scenarios')
    ax.set_ylabel('NPV (€)')
    ax.set_title('NPV Distribution Comparison for Heating-Cooling')
    ax.set_xticks(x_pos)
    # ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    
    # Set y-axis limits
    ax.set_ylim(-20000, 1000)
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add lightcoral background below y=0
    ax.axhspan(ax.get_ylim()[0], 0, alpha=0.2, color='lightcoral', label='Negative NPV region')
    
    # Add legend
    # ax.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Example usage
if __name__ == "__main__":
    # Define scenarios and file paths
    scenario = "S01_simple_shading"

    threshold_S01 = 47.5
    limit = 15

    name = "Heating-Cooling"
    file_path = os.path.join("CBA_", "results", "NPV", f"NPV_H-C_{scenario}.csv")

    output_path = os.path.join("CBA_", "results")
    output_path_1 = os.path.join("CBA_", "results", f"NPV_{scenario}_{name}_markers.png")

    plot_single_npv_distribution(
        file_path,
        scenario,
        output_path=output_path_1,
        threshold_1=threshold_S01,
        bw_method=None,
        limit=limit,
        marker=False,
    )


    """
    scenario_names = [scenario_1, scenario_2, scenario_3]
    file_paths = [file_path_1, file_path_2, file_path_3]


    plot_npv_distribution_bars(
        file_paths=file_paths,
        scenario_names=scenario_names,
        output_path=os.path.join(output_path, f"{name}_NPV_distribution_markers.png"),
        figsize=(12,7)  # Landscape format with larger size
    )

    # Example usage of the new dual regression function
    plot_dual_savings_sum_regression(
        file_path=file_path_1,
        scenario_name=scenario_1,
        output_path=os.path.join(output_path, f"{scenario_1}_dual_regression.png")
    )

    # Example usage of the new overlay regression function
    plot_overlay_savings_sum_regression(
        file_path=file_path_1,
        scenario_name=scenario_1,
        output_path=os.path.join(output_path, f"{scenario_1}_overlay_regression.png")
    )

    scenario_names = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5]
    file_paths = [file_path_1, file_path_2, file_path_3, file_path_4, file_path_5]
    plot_single_npv_distribution(
        file_path_5,
        scenario_5,
        output_path=output_path_5,
        threshold_1=threshold_D01,
        bw_method=None,
        limit=15,
        marker=False,
    )
    plot_npv_distribution_bars(
        file_paths=file_paths,
        scenario_names=scenario_names,
        output_path=os.path.join(output_path, f"{name}_NPV_distribution_markers.png"),
        figsize=(26, 11)  # Landscape format with larger size
    )
    scenario_1 = "Combined"
    scenario_2 = "Heating"
    scenario_3 = "Cooling"
    name = "S04 simple combi"
    file_path_1 = os.path.join("CBA_", "results", "15-06-25", f"NPV_H-C_S04_simple_combi.csv")
    file_path_2 = os.path.join("CBA_", "results", "15-06-25", f"NPV_H_S04_simple_combi.csv")
    file_path_3 = os.path.join("CBA_", "results", "15-06-25", f"NPV_C_S04_simple_combi.csv")

    # Define scenarios and file paths
    scenario_1 = "S01_simple_shading-extra"
    scenario_2 = "S02_simple_paint"
    scenario_3 = "S04_simple_combi"
    name = "Cooling"
    file_path_1 = os.path.join("CBA_", "results", "15-06-25", f"NPV_C_{scenario_1}.csv")
    file_path_2 = os.path.join("CBA_", "results", "15-06-25", f"NPV_C_{scenario_2}.csv")
    file_path_3 = os.path.join("CBA_", "results", "15-06-25", f"NPV_C_{scenario_3}.csv")
    output_path = os.path.join("CBA_", "results", "15-06-25")
    
    output_path_1 = os.path.join("CBA_", "results", "15-06-25", f"NPV_{scenario_1}_{name}.png")
    output_path_2 = os.path.join("CBA_", "results", "15-06-25", f"NPV_{scenario_2}_{name}.png")
    output_path_3 = os.path.join("CBA_", "results", "15-06-25", f"NPV_{scenario_3}_{name}.png")

    # Create boxplot comparison
    print("\nCreating boxplot comparison:")
    plot_npv_boxplots(
        file_paths=[file_path_1, file_path_2, file_path_3],
        scenario_names=[scenario_1, scenario_2, scenario_3],
        output_path=os.path.join(output_path, f"{name}_NPV_boxplot_comparison.png")
    )


        # Example usage of the new function
    slope, intercept, r_squared = analyze_savings_regression(
        file_path=file_path,
        scenario_name=scenario_1,
        variable_key='Cooling',
        savings_threshold=threshold_S01_2
    )

    # Create boxplot comparison
    print("\nCreating boxplot comparison:")
    plot_npv_boxplots(
        file_paths=[file_path_1, file_path_2, file_path_3],
        scenario_names=[scenario_1, scenario_2, scenario_3],
        output_path=os.path.join(output_path, f"{name}_NPV_boxplot_comparison.png")
    ) 

    print("\nAnalyzing Scenario 2:")
    plot_single_npv_distribution(
        file_path_2,
        scenario_2,
        output_path=None,
        color='navy',
        color_hist='lightsteelblue',
        bw_method=0.3,
    )

    print("\nAnalyzing Scenario 3:")
    plot_single_npv_distribution(
        file_path_3,
        scenario_3,
        output_path=None,
        color='navy',
        color_hist='lightsteelblue',
        bw_method=0.3,
    )"""
    
    
