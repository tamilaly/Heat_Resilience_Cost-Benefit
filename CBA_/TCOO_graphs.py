import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde
import scipy.stats as stats
from scipy.interpolate import interp1d

# Import colors from the centralized color configuration
from config.colors import (
    COLOR_MAIN, COLOR_HIST, COLOR_RED, COLOR_GREEN, COLOR_PINK,
    COLOR_1, COLOR_2, COLOR_3
)

# Additional color specific to this file
COLOR_0 = (242/255, 99/255, 102/255)

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

def plot_operational_costs(file_path, scenario_name, marker=False, threshold=None, output_path=None, bw_method=None, cooling=True, heating=False):
    """
    ONLY USE WITH DISCOUNTED COSTS
    Plot operation costs distribution from a single file and optionally save as PNG.

    Args:
        file_path (str): Path to the CSV file containing operation costs data
        scenario_name (str): Name of the scenario for the plot title
        threshold (float, optional): Threshold for the operation costs distribution. If None, no threshold is used.
        bw_method (float, optional): Bandwidth method for KDE. If None, no bandwidth method is used.
        cooling (bool, optional): Whether to plot the cooling operation costs distribution. If True, plot the cooling operation costs distribution.
        heating (bool, optional): Whether to plot the heating operation costs distribution. If True, plot the heating operation costs distribution.
    """
    df = pd.read_csv(file_path)

    if cooling == True:
        df['Cooling_Energy_Cost_List'] = df['Cooling_Cost'].apply(lambda x: [float(i) for i in x.strip('"').split(',')])
    if heating == True:
        df['Heating_Energy_Cost_List'] = df['Heating_Cost'].apply(lambda x: [float(i) for i in x.strip('"').split(',')])

    # sum energy cost list for each row
    # first create an empty column operational costs
    df['Operational_Costs'] = 0

    if cooling == True:
        df['Operational_Costs'] += df['Cooling_Energy_Cost_List'].apply(lambda x: sum(x))
    if heating == True:
        df['Operational_Costs'] += df['Heating_Energy_Cost_List'].apply(lambda x: sum(x))

    oc_list = df["Operational_Costs"].values
    mean_oc = np.mean(oc_list)
    
    # Gaussian KDE
    if bw_method == None:
        kde = gaussian_kde(oc_list)
    else:
        kde = gaussian_kde(oc_list, bw_method=bw_method)

    x_vals = np.linspace(min(oc_list), max(oc_list), 1000)
    y_vals = kde(x_vals)

    lower_bound = min(oc_list)
    upper_bound = max(oc_list)

    if marker == True:
        if cooling == True:
            # Count energy costs above threshold for each row
            df['Cooling_Energy_Cost_Above_Threshold'] = df['Cooling_Energy_Cost_List'].apply(lambda x: sum(1 for i in x if i >= threshold))
            # find the mean of the operational costs for the rows where the cooling energy cost is above threshold
            limit = 8
            oc_values_at_limit = df['Operational_Costs'][df['Cooling_Energy_Cost_Above_Threshold'] == limit]
            limit_up = np.mean(oc_values_at_limit)
            percentile_values = find_percentiles(kde, x_vals, [limit_up])
            prob_up = 1 - percentile_values[0]
            print(f"Probability of {limit_up} Euro or more: {prob_up*100:.0f}%")
    

    # Plot the TCOO distribution
    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_vals, color=COLOR_MAIN, label="KDE")
    plt.hist(df["Operational_Costs"], bins=100, color=COLOR_HIST, alpha=0.5, label="Histogram", density=True)

    # Add vertical line for mean
    plt.axvline(x=mean_oc, color=COLOR_MAIN, linestyle='--', alpha=0.7, label=f'Mean: {mean_oc:,.0f} €')

    if marker == True:
        # plot the marker with axv line
        plt.axvline(x=limit_up, color=COLOR_RED, linestyle=':', label=f'{limit} or more extreme years')
        # fill the middle
        mask_3 = (x_vals <= limit_up)
        plt.fill_between(x_vals[mask_3], 0, y_vals[mask_3], color=COLOR_GREEN, alpha=0.3, label=f'mild to average outcomes')
        prob_mid = percentile_values[0]
        place= (limit_up - lower_bound)/2 + lower_bound
        plt.text(place, 0.00005, f'{prob_mid*100:.0f}%', color=COLOR_GREEN, fontsize=12)

        # shade the area above the curve from marker to max
        mask = x_vals >= limit_up
        plt.fill_between(x_vals[mask], 0, y_vals[mask], color=COLOR_RED, alpha=0.3)
        place= limit_up + (upper_bound- limit_up)/2
        plt.text(place, 0.00005, f'{prob_up*100:.0f}%', color=COLOR_RED, fontsize=12)

    plt.title(f"Operation Costs Distribution for {scenario_name}")
    plt.xlabel("Operation Costs (€)")
    plt.ylabel("Frequency")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.legend()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    

def plot_single_tcoo_distribution(file_path, scenario_name, marker=False, threshold=None, output_path=None, bw_method=None):
    """
    Plot TCOO distribution from a single file and optionally save as PNG.
    
    Args:
        file_path (str): Path to the CSV file containing TCOO data
        scenario_name (str): Name of the scenario for the plot title
        threshold (float, optional): Threshold for the TCOO distribution. If None, no threshold is used.
        output_path (str, optional): Path to save the PNG file. If None, plot is only displayed
        bw_method (float): Bandwidth method for KDE
    """
    df = pd.read_csv(file_path)
    
    if marker == True:
        # Parse the energy cost string into a list of numbers for each row
        df['Energy_Cost_List'] = df['Cooling_Cost'].apply(lambda x: [float(i) for i in x.strip('"').split(',')])

        # Count energy costs above threshold for each row
        df['Energy_Cost_Above_Threshold'] = df['Energy_Cost_List'].apply(lambda x: sum(1 for i in x if i >= threshold))
        # df['Energy_Cost_Below_Threshold_2'] = df['Energy_Cost_List'].apply(lambda x: sum(1 for i in x if i <= threshold_2))

        # Get TCOO values
        TCOO_list = df['TCOO'].values
        energy_costs_above_threshold = df['Energy_Cost_Above_Threshold'].values
        # energy_costs_below_threshold_2 = df['Energy_Cost_Below_Threshold_2'].values
        # Fit linear regression threshold 1
        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(TCOO_list, energy_costs_above_threshold)
        line_1 = slope_1 * TCOO_list + intercept_1

        # Fit linear regression threshold 2
        # slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(TCOO_list, energy_costs_below_threshold_2)
        # line_2 = slope_2 * TCOO_list + intercept_2
        """
        # Plot first regression (above threshold)
        plt.figure(figsize=(12, 6))
        plt.scatter(TCOO_list, energy_costs_above_threshold, color=COLOR_RED, label='Data points')
        plt.plot(TCOO_list, line_1, color=COLOR_MAIN, label='Linear fit')
        plt.xlabel('TCOO')
        plt.ylabel(f'Energy costs above {threshold}€')
        plt.legend()
        plt.savefig(os.path.join('CBA_', 'test_code', f"{scenario_name}_TCOO_limit_upper_regression.png"), dpi=300, bbox_inches='tight')
        plt.close()
        """
        # find the TCOO for the line intersection at y=8
        limit = 8
        # tcoo values with limit
        tcoo_values_at_limit = TCOO_list[energy_costs_above_threshold == limit]
        # tcoo_values_at_limit_2 = TCOO_list[energy_costs_below_threshold_2 == limit]
        
        # print(f"Maximum TCOO value with {limit} years: {np.mean(tcoo_values_at_limit_2):,.0f} Euro")
        limit_up = np.mean(tcoo_values_at_limit) # round((limit - intercept) / slope)
        # limit_down = np.mean(tcoo_values_at_limit_2)
    else:        
        # Parse the energy cost string into a list of numbers for each row
        df['Energy_Cost_List'] = df['Cooling_Cost'].apply(lambda x: [float(i) for i in x.strip('"').split(',')])

        # Count energy costs above threshold for each row
        df['Energy_Cost_Above_Threshold'] = df['Energy_Cost_List'].apply(lambda x: sum(1 for i in x if i >= threshold))
        # df['Energy_Cost_Below_Threshold_2'] = df['Energy_Cost_List'].apply(lambda x: sum(1 for i in x if i <= threshold_2))

        # Get TCOO values
        TCOO_list = df['TCOO'].values
        energy_costs_above_threshold = df['Energy_Cost_Above_Threshold'].values

        # find the TCOO for the line intersection at y=8
        limit = 8
        # tcoo values with limit
        tcoo_values_above_limit = TCOO_list[energy_costs_above_threshold >= limit]

    # Gaussian KDE
    if bw_method == None:
        kde = gaussian_kde(df["TCOO"])
    else:
        kde = gaussian_kde(df["TCOO"], bw_method=bw_method)

    x_vals = np.linspace(min(df["TCOO"]), max(df["TCOO"]), 1000)
    y_vals = kde(x_vals)
    
    if marker == True:
        # find percentiles at limit_up and limit_down
        percentile_values = find_percentiles(kde, x_vals, [limit_up])

        # prob_down = percentile_values[0]
        prob_up = 1 - percentile_values[0]
        # print(f"Probability of {limit_down} Euro or less: {prob_down*100:.0f}%")
    

    # Calculate the mean and standard deviation of the TCOO distribution
    mean_tcoo = df["TCOO"].mean()
    std_tcoo = df["TCOO"].std()

    # Central 95% interval of the TCOO distribution
    lower_bound = np.percentile(df["TCOO"], 2.5)
    upper_bound = np.percentile(df["TCOO"], 97.5)
    eighty_percentile = np.percentile(df["TCOO"], 80)

    # Calculate the proportion of values within the confidence interval
    values_in_ci = df["TCOO"][(df["TCOO"] >= lower_bound) & (df["TCOO"] <= upper_bound)]
    total_values = len(df["TCOO"])
    ci_proportion = len(values_in_ci) / total_values
    ninety_ninth_percentile = np.percentile(df["TCOO"], 99)

    # Print the mean and confidence intervals
    print(f"TCOO for {scenario_name}")
    print(f"Mean TCOO: {mean_tcoo:.2f} Eur")
    print(f"Standard Deviation: {std_tcoo:.2f} Eur")
    print(f"95% CI: {lower_bound:.2f} to {upper_bound:.2f} Eur")
    if marker == True:
        print(f"Minimum TCOO values with {limit} years: {np.mean(tcoo_values_at_limit):,.0f} Euro")
        print(f"Probability of {round(limit_up, 0)} Euro or more: {prob_up*100:.0f}%")
    print(f"99th percentile: {ninety_ninth_percentile:.2f} Eur")

    # Plot the TCOO distribution
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, y_vals, color=COLOR_MAIN, label="KDE")
    plt.hist(df["TCOO"], bins=100, color=COLOR_HIST, alpha=0.5, label="Histogram", density=True)  
    
    if marker == True:
        # Add vertical line for mean
        plt.axvline(x=mean_tcoo, color=COLOR_GREEN, linestyle='--', alpha=0.7, label=f'Mean: {mean_tcoo:,.0f} €')
        # fill the middle with hist color
        mask_3 = (x_vals <= limit_up)
        plt.fill_between(x_vals[mask_3], 0, y_vals[mask_3], color=COLOR_GREEN, alpha=0.3, label=f'mild to average outcomes')
        prob_mid = 1-prob_up
        place= (limit_up - lower_bound)/2 + lower_bound
        plt.text(place, 0.00005, f'{prob_mid*100:.0f}%', color=COLOR_GREEN, fontsize=12)

        # plot the marker with axv line
        plt.axvline(x=limit_up, color=COLOR_PINK, linestyle='--', label=f'Threshold: {limit_up:,.0f} €')
        # shade the area above the curve from marker to max
        mask = x_vals >= limit_up
        plt.fill_between(x_vals[mask], 0, y_vals[mask], color=COLOR_PINK, alpha=0.3, label=f'{limit} or more extreme years')
        place= limit_up + (upper_bound- limit_up)/2
        plt.text(place, 0.00005, f'{prob_up*100:.0f}%', color=COLOR_PINK, fontsize=12)
        # plot the 99th percentile
        plt.axvline(x=ninety_ninth_percentile, color=COLOR_RED, linestyle='--', label=f'99th Percentile: {ninety_ninth_percentile:,.0f} €')
    else:
        # Add vertical line for mean
        plt.axvline(x=mean_tcoo, color=COLOR_MAIN, linestyle='--', alpha=0.7, label=f'Mean: {mean_tcoo:,.0f} €')
        if len(tcoo_values_above_limit) > 0:
            for i in range(len(tcoo_values_above_limit)):
                plt.axvline(x=tcoo_values_above_limit[i], color=COLOR_PINK, linestyle=':', linewidth=0.8, 
                           label=f'TCOO values above {limit} years' if i==0 else '')

        # plt.axvline(x=ninety_ninth_percentile, color=COLOR_MAIN, linestyle='--', alpha=0.7, label=f'99th P: {ninety_ninth_percentile:,.0f} €')

    # scenario names remove _
    scenario_name = scenario_name.replace("_", " ")

    plt.title(f"TCOO Distribution for {scenario_name}")
    plt.xlabel("Total Cost of Ownership (€)")
    plt.ylabel("Frequency")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_compare_tcoo_distributions(file_path_1, file_path_2, scenario_name, output_path=None, 
                                 color1='red', color2='navy', 
                                 color_hist1='lightcoral', color_hist2='lightsteelblue',
                                 bw_method1=None, bw_method2=None):
    """
    Plot and compare TCOO distributions from two files and optionally save as PNG.
    
    Args:
        file_path_1 (str): Path to the first CSV file containing TCOO data
        file_path_2 (str): Path to the second CSV file containing TCOO data
        scenario_name (str): Name of the scenario for the plot title
        output_path (str, optional): Path to save the PNG file. If None, plot is only displayed
        color1 (str): Color for the first KDE line
        color2 (str): Color for the second KDE line
        color_hist1 (str): Color for the first histogram
        color_hist2 (str): Color for the second histogram
        bw_method1 (float): Bandwidth method for first KDE
        bw_method2 (float): Bandwidth method for second KDE
    """
    df = pd.read_csv(file_path_1)
    df_2 = pd.read_csv(file_path_2)
    
    # Gaussian KDE for first file
    if bw_method1 == None:
        kde = gaussian_kde(df["TCOO"])
    else:
        kde = gaussian_kde(df["TCOO"], bw_method=bw_method1)
    x_vals = np.linspace(min(df["TCOO"]), max(df["TCOO"]), 1000)
    y_vals = kde(x_vals)
    
    # Gaussian KDE for second file
    if bw_method2 == None:
        kde_2 = gaussian_kde(df_2["TCOO"])
    else:
        kde_2 = gaussian_kde(df_2["TCOO"], bw_method=bw_method2)
    x_vals_2 = np.linspace(min(df_2["TCOO"]), max(df_2["TCOO"]), 1000)
    y_vals_2 = kde_2(x_vals_2)
    
    # Plot the TCOO distributions
    plt.figure(figsize=(12, 3))
    plt.plot(x_vals, y_vals, color=color1, label="KDE (from Gaussian PDF)")
    plt.plot(x_vals_2, y_vals_2, color=color2, label="KDE (from Normal PDF)")
    
    # normalize histograms
    plt.hist(df["TCOO"], bins=100, color=color_hist1, alpha=0.5, label="Histogram (Gaussian PDF)", density=True)
    plt.hist(df_2["TCOO"], bins=100, color=color_hist2, alpha=0.5, label="Histogram (Normal PDF)", density=True)
    plt.title(f"TCOO Distribution for {scenario_name}")
    plt.xlabel("TCOO")
    plt.ylabel("Frequency")
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_tcoo_boxplots(file_paths, scenario_names, output_path=None, figsize=(6, 8)):
    """
    Create boxplots comparing TCOO across different scenarios.
    
    Args:
        file_paths (list): List of paths to CSV files containing TCOO data for each scenario
        scenario_names (list): List of scenario names corresponding to each file
        output_path (str, optional): Path to save the PNG file. If None, plot is only displayed
        figsize (tuple): Figure size as (width, height) in inches
    """
    # Read all data files
    data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data.append(df["TCOO"])
    
    # Create the boxplot
    plt.figure(figsize=figsize)
    boxplot = plt.boxplot(data, labels=scenario_names, patch_artist=True, showfliers=False)
    # Customize boxplot colors
    colors = [COLOR_0, COLOR_1, COLOR_2, COLOR_3]
    for patch, color in zip(boxplot['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
    
    # Set median line color to black
    for median in boxplot['medians']:
        median.set_color('black')
    
    # Add labels and title
    plt.ylabel('TCOO')
    plt.title('TCOO Distribution Comparison for Cooling')
    
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

def plot_tcoo_vs_energy_cost(file_path, output_path, key='Cooling', threshold=550, color=COLOR_MAIN):
    """
    Create a linear regression plot comparing TCOO against energy costs above threshold.
    
    Args:
        file_path (str): Path to the CSV file containing TCOO and energy cost data
        output_path (str): Path to save the output PNG file
        threshold (float): Energy cost threshold in Euro (default: 550)
        figsize (tuple): Figure size as (width, height) in inches
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Parse the energy cost string into a list of numbers for each row
    df['Energy_Cost_List'] = df[key+'_Cost'].apply(lambda x: [float(i) for i in x.strip('"').split(',')])
    
    # Count energy costs above threshold for each row
    df['Energy_Cost_Above_Threshold'] = df['Energy_Cost_List'].apply(lambda x: sum(1 for i in x if i > threshold))

    # Get TCOO values
    TCOO_list = df['TCOO'].values
    energy_costs_above_threshold = df['Energy_Cost_Above_Threshold'].values

    limit=8
    tcoo_values_at_limit = TCOO_list[energy_costs_above_threshold == limit]
    mean_tcoo_at_limit = np.mean(tcoo_values_at_limit)

    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(TCOO_list, energy_costs_above_threshold)
    line = slope * TCOO_list + intercept
    print(f"R-squared: {r_value**2:.3f}")

    # Create scatterplot
    plt.figure(figsize=(14, 8))
    plt.scatter(TCOO_list, energy_costs_above_threshold, color=color, label='Data points')
    plt.scatter(mean_tcoo_at_limit, limit, color='black', label=f'Mean TCOO at {limit} years: {mean_tcoo_at_limit:,.0f} €')
    plt.plot(TCOO_list, line, color=COLOR_RED, label=f'Linear fit (y = {slope:.3f}x + {intercept:.3f})')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.axhline(y=limit, color='black', linestyle='--')
    plt.xlabel('TCOO (€)')
    plt.ylim(0, 15)
    plt.ylabel(f'Annual {key} costs above threshold (€{threshold})')
    plt.title(f'TCOO vs Annual {key} costs above threshold: {threshold} €')
    plt.legend()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Drop the temporary columns before saving
    df = df.drop(['Energy_Cost_List'], axis=1)
    
    # Save the modified dataframe back to CSV
    df.to_csv(file_path, index=False)

# Example usage
if __name__ == "__main__":
    # Define scenarios and file paths
    scenario_0 = "baseline"
    scenario_1 = "S01_shading"
    scenario_2 = "S02_paint"
    scenario_3 = "S04_combi"
    scenario_4 = "M01_medium_shading"
    scenario_5 = "D01_deep_shading-paint"

    threshold_universal = 1000
    threshold_baseline = 1115 
    threshold_S01 = 1059 # 1053
    threshold_S02 = 1086
    threshold_S04 = 1033
    threshold_M01 = 1054
    threshold_D01 = 1030


    base_2020 = round((17515.24*0.243)/2.9, 2)
    name = f"Baseline"
    file_path_0 = r"CBA_\results\15-06-25\TCOO_baseline_C.csv"
    file_path_1 = r"CBA_\results\15-06-25\TCOO_S01_simple_shading-extra_C.csv"
    file_path_2 = r"CBA_\results\15-06-25\TCOO_S02_simple_paint_C.csv"
    file_path_3 = r"CBA_\results\15-06-25\TCOO_S04_simple_combi_C.csv"
    file_path_4 = r"CBA_\results\15-06-25\TCOO_M01_medium_shading_C.csv"
    file_path_5 = r"CBA_\results\15-06-25\TCOO_D01_deep_shading-paint_C.csv"

    output_path = r"CBA_\results\15-06-25"
    
    plot_single_tcoo_distribution(
        file_path=file_path_0,
        scenario_name=scenario_0,
        marker=False,
        threshold=threshold_baseline,
        output_path=os.path.join(output_path, f"{scenario_0}_TCOO_distribution_cooling_markers.png")
    )



"""
    plot_tcoo_vs_energy_cost(
        file_path=file_path_1,
        output_path=os.path.join(output_path, f"{scenario_1}_TCOO_vs_energy_cost.png"),
        threshold=threshold_S01,
        color=COLOR_0
    )




        plot_operational_costs(
        file_path=file_path_0,
        scenario_name=scenario_0,
        marker=True,
        threshold=threshold_baseline,
        output_path=os.path.join(output_path, f"{scenario_0}_OC_distribution_cooling.png"),
        cooling=True,
        heating=False
    )

Baseline
TCOO for 16 Eur: 14914.00 Eur
Mean TCOO: 14383.66 Eur
Standard Deviation: 1426.91 Eur
95% CI: 11656.05 to 17166.39 Eur
Values within CI: 95.00%
S01
TCOO for 16 Eur: 24001.00 Eur
Mean TCOO: 23502.15 Eur
Standard Deviation: 1368.66 Eur
95% CI: 20872.15 to 26043.14 Eur
Values within CI: 95.00%
S02
TCOO for 16 Eur: 24362.00 Eur
Mean TCOO: 23981.61 Eur
Standard Deviation: 1367.98 Eur
95% CI: 21416.59 to 26645.80 Eur
Values within CI: 95.00%
S04
TCOO for 16 Eur: 33364.00 Eur
Mean TCOO: 32870.65 Eur
Standard Deviation: 1345.09 Eur
95% CI: 30233.87 to 35646.26 Eur
Values within CI: 95.00%
"""


"""
    # Create boxplot comparison
    print("\nCreating boxplot comparison:")
    plot_tcoo_boxplots(
        file_paths=[file_path, file_path_1, file_path_2, file_path_3],
        scenario_names=[scenario_0, scenario_1, scenario_2, scenario_3],
        output_path=os.path.join(output_path, f"{name}_TCOO_boxplot_comparison.png")
    ) 
    # Create the TCOO vs Energy Cost plot
    plot_tcoo_vs_energy_cost(
        file_path=file_path,
        output_path=os.path.join(output_path, f"Baseline_{name}.png"),
        threshold=base_2020,
        color=COLOR_0
    )

"""
    
"""   
    # Create boxplot comparison
    print("\nCreating boxplot comparison:")
    plot_npv_boxplots(
        file_paths=[file_path_1, file_path_2, file_path_3],
        scenario_names=[scenario_1, scenario_2, scenario_3],
        output_path=os.path.join(output_path, f"{name}_NPV_boxplot_comparison.png")
    ) 

    # Run single file analysis for each scenario
    print("\nAnalyzing Scenario 1:")
    plot_single_npv_distribution(
        file_path_1,
        scenario_1,
        output_path=None,
        color='navy',
        color_hist='lightsteelblue',
        bw_method=0.3,
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
    
    
