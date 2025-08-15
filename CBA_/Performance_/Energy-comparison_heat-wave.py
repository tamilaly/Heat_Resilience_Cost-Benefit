import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import os
import numpy as np
from scipy import stats

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define colors - RGB values normalized to 0-1 range
r, g, b = 112/255, 44/255, 168/255
r, g, b = 5/255, 0/255, 113/255
# r, g, b = 204/255, 31/255, 4/255
COLOR_MAIN = (r, g, b) # 'darkorange'
COLOR_HIST = tuple(0.5 * (c + 1) for c in COLOR_MAIN)
# COLOR_MAIN = 'darkorange'
COLOR_0 = 242/255, 99/255, 102/255
COLOR_1 = (243/255, 182/255, 68/255)
COLOR_2 = 'darkorange'
COLOR_3 = (98/255, 177/255, 163/255)
COLOR_4 = (237/255, 145/255, 185/255)
COLOR_5 = (5/255, 0/255, 113/255)

def reconstruct_pdf_from_percentiles(csv_file,
                                     scenario_name,
                                     output_column='Cooling_Annual_Total',
                                     cop=None,
                                     percentiles=[0, 5, 25, 50, 75, 95, 100], 
                                     num_samples=1000, 
                                     plot=True,
                                     color_main='purple',
                                     color_hist='mediumpurple'):
    """
    Reconstruct a probability density function from a limited set of percentiles.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing the data
    output_column : str
        Column name to use for the PDF reconstruction
    percentiles : list
        List of percentiles corresponding to the data points
    num_samples : int
        Number of samples to generate for the reconstructed PDF
    plot : bool
        Whether to plot the reconstructed PDF
        
    Returns:
    --------
    samples : numpy.ndarray
        Generated samples from the reconstructed PDF
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Sort df by the output column in ascending order
    sorted_df = df.sort_values(by=output_column)
    sorted_values = sorted_df[output_column].values
       
    # apply cop to the sorted values
    sorted_values = sorted_values / cop
    # Sort df by the output column in ascending order
    sorted_years = sorted_df['Year'].values
    
    # Check if we have the right number of percentiles
    if len(sorted_values) != len(percentiles):
        print(f"Warning: Number of data points ({len(sorted_values)}) doesn't match number of percentiles ({len(percentiles)})")
        # Adjust percentiles if needed
        if len(sorted_values) < len(percentiles):
            percentiles = percentiles[:len(sorted_values)]
        else:
            # If we have more data points than percentiles, we need to estimate the percentiles
            print("Estimating percentiles from data...")
            percentiles = [i * 100 / (len(sorted_values) - 1) for i in range(len(sorted_values))]
    
    # Convert percentiles to probabilities (0-1 range)
    probabilities = np.array(percentiles) / 100.0
    
    # Method 1: Interpolation between percentiles
    # Create an interpolation function
    interp_func = interp1d(probabilities, sorted_values, bounds_error=False, fill_value="extrapolate")

    # Generate uniform random samples
    uniform_samples = np.random.uniform(0, 1, num_samples)
    
    # Transform uniform samples using the inverse CDF (interpolation function)
    samples = interp_func(uniform_samples)
    
    # This is an alternative approach that might give smoother results
    kde = stats.gaussian_kde(samples, bw_method=0.6) # , bw_method=0.6

    kde_samples = kde.resample(num_samples)[0]
    kde_samples = kde_samples[kde_samples >= 0]  # Remove negative values

    min_bound = min(sorted_values)
    max_bound = max(sorted_values)

    kde_samples = kde_samples[kde_samples >= min_bound]
    kde_samples = kde_samples[kde_samples <= max_bound]

    if plot:
        plt.figure(figsize=(12, 8))
        # Plot the original data points
        plt.scatter(sorted_values, probabilities, color=color_main, s=10, label='year values')
        # label years on the plot
        for i, (value, year) in enumerate(zip(sorted_values, sorted_years)):
            if year == 2020:
                plt.text(value*1.01, probabilities[i]*0.98, str(year), color='tab:red', fontsize=10)
                plt.scatter(value, probabilities[i], color='tab:red', s=25)

        # Plot the interpolation function
        x_interp = np.linspace(0, 1, 1000)
        y_interp = interp_func(x_interp)
        plt.plot(y_interp, x_interp, color_main, alpha=0.5, label='Interpolation')
        plt.xlabel("Cooling Energy Use (kWh per year)")
        plt.ylabel('Cumulative Probability (%)')
        plt.title(f'{scenario_name.capitalize()} CDF Energy Use')
        plt.legend()
        plt.grid(True, axis='y')
        plt.show()
        # Plot the KDE curve
        plt.figure(figsize=(12, 8))
        # Plot histograms of the generated samples
        # plt.hist(kde_samples, bins=50, density=True, alpha=0.3, color=color_hist, label='KDE Samples')
        
        # Plot the KDE curve
        x_kde = np.linspace(min(samples), max(samples), 1000)
        plt.plot(x_kde, kde(x_kde), color_main, label='KDE')
        
        # Add vertical lines for 2020 and 2007 values
        value_2020 = sorted_values[sorted_years == 2020][0]
        value_2007 = sorted_values[sorted_years == 2007][0]
        plt.axvline(x=value_2020, color='tab:red', linestyle=':', label='min & max values')
        plt.scatter(value_2020, kde(value_2020), color='tab:red', s=35, label='2020')
        plt.axvline(x=value_2007, color='tab:red', linestyle=':')

        plt.xlabel("Cooling Energy Use (kWh per year)")
        plt.ylabel('Probability')
        plt.title(f'{scenario_name}: Reconstructed Probability Density Function')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return samples, kde_samples, interp_func, kde
def compare_heatwave_impact(base_file, heatwave_file, years=[2020], cop=2.9, key='cooling'or'overheating', price_electricity=0.243):
    """
    Compare the impact of heatwave on 2020
    Make a plot showing the percentage of energy use in 2020 during heat wave
    """

    # load heatwave file
    df_heatwave = pd.read_csv(heatwave_file)

    base_df = pd.read_csv(base_file)

    # search for years in base_df
    base_df = base_df[base_df['Year'].isin(years)]
    df_heatwave = df_heatwave[df_heatwave['Year'].isin(years)]

    # sort by year
    base_df = base_df.sort_values(by='Year')
    df_heatwave = df_heatwave.sort_values(by='Year')

    if key == 'cooling':
        # get cooling energy use in 2020
        full_values = base_df['Cooling_Annual_Total'].values
        # get cooling energy use in heatwave
        heatwave_values = df_heatwave['Cooling_Energy_Demand'].values
        
        # Get COP and divide energy values
        # full_values = full_values / cop
        # heatwave_values = heatwave_values / cop
        
        # calculate percentage
        percentage = (heatwave_values / full_values) * 100
        for i, year in zip(percentage, years):
            print(f"The heatwave of {year} accounted for {i:.2f}% of cooling energy use")
    
    if key == 'energy':
        # get heating energy use in 2020
        full_values_heating = base_df['Heating_Annual_Total'].values
        full_values_cooling = base_df['Cooling_Annual_Total'].values
        # get heating energy use in heatwave
        heatwave_values_cooling = df_heatwave['Cooling_Energy_Demand'].values
        
        # Get COP and divide energy values
        full_values_heating = full_values_heating / 0.85
        full_values_cooling = full_values_cooling / cop
        heatwave_values_cooling = heatwave_values_cooling / cop

        # add heating and cooling energy use heating and cooling energy use
        full_values = full_values_heating + full_values_cooling
        heatwave_values = heatwave_values_cooling


        # calculate percentage
        percentage = (heatwave_values / full_values) * 100
        for i in range(len(years)):
            print(f"The heatwave of {years[i]} accounted for {percentage[i]:.2f}% of energy use ({heatwave_values[i]:.0f} kWh) ({full_values[i]:.0f} kWh)")

    elif key == 'cost':
        # get cooling energy use in 2020
        full_values = base_df['Cooling_Annual_Total'].values
        # get cooling energy use in heatwave
        heatwave_values = df_heatwave['Cooling_Energy_Demand'].values
        
        # Get COP and divide energy values
        full_values = (full_values / cop)*price_electricity
        heatwave_values = (heatwave_values / cop)*price_electricity
        
        # calculate percentage
        percentage = (heatwave_values / full_values) * 100
        for i, year in zip(percentage, years):
            print(f"The heatwave of {year} accounted for {i:.2f}% of cooling energy cost")        

    elif key == 'overheating':
        # get overheating for the scenario
        full_values = base_df['Overheated_Hours_28_SET'].values
        # get overheating for the scenario in heatwave
        heatwave_values = df_heatwave['Overheated_Hours_28_SET'].values
        # calculate percentage
        percentage = (heatwave_values / full_values) * 100
        for i in percentage:
            print(f"The heatwave of 2020 accounted for {i:.2f}% of overheating considering")
        

    # Create DataFrame
    df_comparison = pd.DataFrame({
        'Year': years,
        'Annual': full_values,
        'Heatwave': heatwave_values
    })
    
    return df_comparison
def plot_heatwave_comparison(df_comparison, key='cooling'):
    """
    Create an overlaid bar chart comparing full 2020 values with heatwave values for each scenario
    """
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    barWidth = 0.4
    
    # Set positions of the bars on X axis
    r = np.arange(len(df_comparison['Year']))
    
    # Create the bars - using same color with different opacity
    plt.bar(r, df_comparison['Annual'], width=barWidth, color=COLOR_MAIN, label='Annual', alpha=0.6)
    plt.bar(r, df_comparison['Heatwave'], width=barWidth, color=COLOR_MAIN, label='Heatwave', alpha=1.0)
    
    # Calculate and add percentage labels
    for i in range(len(df_comparison)):
        percentage = (df_comparison['Heatwave'][i] / df_comparison['Annual'][i]) * 100
        plt.text(r[i], df_comparison['Heatwave'][i], f'{percentage:.1f}%', 
                ha='center', va='top', fontsize=10, color='white')
        plt.text(r[i], df_comparison['Heatwave'][i], f'{df_comparison["Heatwave"][i]:.0f}', 
                ha='center', va='bottom', fontsize=10, color=COLOR_MAIN)
        plt.text(r[i], df_comparison['Annual'][i], f'{df_comparison["Annual"][i]:.0f}', 
                ha='center', va='bottom', fontsize=10, color=COLOR_MAIN)
    
    # Add labels and title
    if key == 'cooling':
        plt.ylabel('Cooling Energy Demand (kWh)')#
        plt.title(f'{key.capitalize()} Energy Demand Heatwave Contribution to the Extremes')
    elif key == 'energy':
        plt.ylabel('Energy Use (kWh)')#
        plt.title(f'{key.capitalize()} Energy Use Heatwave Contribution to the Extremes')
    elif key == 'cost':
        plt.ylabel('Cooling Energy Cost (€)')#
        plt.title(f'{key.capitalize()} Cooling Energy Cost Heatwave Contribution to the Extremes')
    elif key == 'overheating':
        plt.ylabel('Overheated Hours (h)') # Degree Hours >28°C SET
        plt.title(f'{key.capitalize()} Heatwave Contribution to the Extremes')
    
    # Add scenario labels
    plt.xticks(r, df_comparison['Year'], rotation=45, ha='right')
    
    # Add legend to bottom left corner
    plt.legend(loc='lower left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.show()

def stats_from_pdf(scenario_name, samples):

    mean_pdf = np.mean(samples)
    std_pdf = np.std(samples)

    print(f"\nDistribution Statistics: {scenario_name} Energy Use")
    # Mean
    print(f"Mean PDF: {mean_pdf:.2f}")
    # Standard Deviation
    print(f"Standard Deviation: {std_pdf:.2f}")
    # COnfidence Intervall
    print(f"95% Confidence Interval: [{mean_pdf - 1.96*std_pdf:.2f}, {mean_pdf + 1.96*std_pdf:.2f}]")
    # print min and max
    print(f"Min: {np.min(samples):.2f}; Max: {np.max(samples):.2f}")

    return mean_pdf, std_pdf

def compare_heatwave_impact_multiple_scenarios(scenario_names, scenario_configs, years=[2020], key='cooling', price_electricity=0.243):
    """
    Compare the impact of heatwave on multiple scenarios for specified years
    Make a plot showing the percentage of energy use during heat wave for each scenario
    """
    
    # Dictionary to store results for each scenario
    scenario_results = {}
    
    # Load the comprehensive heatwave file
    heatwave_file = os.path.join(current_dir, '..', '03-06-25', 'Heatwave_2020.csv')
    try:
        heatwave_df = pd.read_csv(heatwave_file)
    except FileNotFoundError as e:
        print(f"Error: Could not load heatwave file: {e}")
        return {}
    
    for scenario_name in scenario_names:
        print(f"\n=== Analyzing {scenario_name} ===")
        
        # Get scenario configuration
        if scenario_name == "baseline":
            scenario_config = scenario_configs['baseline']
        else:
            scenario_config = scenario_configs[scenario_name]
        
        # Construct base file path
        if key == 'cooling':
            base_file = os.path.join(current_dir, 'scenarios', scenario_config['cooling_file'])
            cop = scenario_config['cooling_cop_or_eer']
        elif key == 'heating':
            base_file = os.path.join(current_dir, 'scenarios', scenario_config['heating_file'])
            cop = scenario_config['heating_cop_or_eer']
        elif key == 'overheating':
            base_file = os.path.join(current_dir, 'scenarios', scenario_config['overheating_file'])
            cop = None  # Overheating doesn't use COP
        else:
            raise ValueError(f"Unsupported key: {key}")
        
        # Load base data
        try:
            base_df = pd.read_csv(base_file)
        except FileNotFoundError as e:
            print(f"Warning: Could not load base data for {scenario_name}: {e}")
            continue
        
        # Filter for specified years
        base_df = base_df[base_df['scenario'].str.lower() == scenario_name.lower()]
        
        # Get heatwave data for this specific scenario
        scenario_heatwave_data = heatwave_df[heatwave_df['scenario'].str.lower() == scenario_name.lower()]
        
        if scenario_heatwave_data.empty:
            print(f"Warning: No heatwave data found for scenario {scenario_name}")
            continue
        
        
        if key == 'cooling':
            # Get cooling energy use
            full_values = base_df['Cooling_Annual_Total'].values
            heatwave_values = scenario_heatwave_data['Energy_Demand'].values
            
            # Calculate percentage
            percentage = (heatwave_values / full_values) * 100
            for i, year in zip(percentage, years):
                print(f"The heatwave of {year} accounted for {i:.2f}% of cooling energy use")
        
        elif key == 'energy':
            # Get heating and cooling energy use
            full_values_heating = base_df['Heating_Annual_Total'].values
            full_values_cooling = base_df['Cooling_Annual_Total'].values
            heatwave_values_cooling = scenario_heatwave_data['Energy_Demand'].values
            
            # Apply COP
            full_values_heating = full_values_heating / 0.85
            full_values_cooling = full_values_cooling / cop
            heatwave_values_cooling = heatwave_values_cooling / cop
            
            # Total energy
            full_values = full_values_heating + full_values_cooling
            heatwave_values = heatwave_values_cooling
            
            # Calculate percentage
            percentage = (heatwave_values / full_values) * 100
            for i in range(len(years)):
                print(f"The heatwave of {years[i]} accounted for {percentage[i]:.2f}% of energy use ({heatwave_values[i]:.0f} kWh) ({full_values[i]:.0f} kWh)")
        
        elif key == 'cost':
            # Get cooling energy use
            full_values = base_df['Cooling_Annual_Total'].values
            heatwave_values = scenario_heatwave_data['Energy_Demand'].values
            
            # Apply COP and price
            full_values = (full_values / cop) * price_electricity
            heatwave_values = (heatwave_values / cop) * price_electricity
            
            # Calculate percentage
            percentage = (heatwave_values / full_values) * 100
            for i, year in zip(percentage, years):
                print(f"The heatwave of {year} accounted for {i:.2f}% of cooling energy cost")
        
        elif key == 'overheating':
            # Get overheating values
            full_values = base_df['Overheated_Hours_28_SET'].values
            heatwave_values = scenario_heatwave_data['Overheated_Hours_28_SET'].values
            
            # Calculate percentage
            percentage = (heatwave_values / full_values) * 100
            for i in percentage:
                print(f"The heatwave of 2020 accounted for {i:.2f}% of overheating")
        
        # Store results
        scenario_results[scenario_name] = {
            'Year': years,
            'Annual': full_values,
            'Heatwave': heatwave_values,
            'Percentage': percentage
        }
    
    return scenario_results

def plot_heatwave_comparison_multiple_scenarios(scenario_results, key='cooling'):
    """
    Create a bar chart comparing heat wave impacts across multiple scenarios
    """
    plt.figure(figsize=(14, 10))
    
    # Get scenario names and their data
    scenario_names = list(scenario_results.keys())
    
    # Extract heatwave values and percentages for plotting
    heatwave_values = []
    percentages = []
    annual_values = []
    
    for scenario_name in scenario_names:
        scenario_data = scenario_results[scenario_name]
        heatwave_values.append(scenario_data['Heatwave'][0])  # Take first year (2020)
        percentages.append(scenario_data['Percentage'][0])
        annual_values.append(scenario_data['Annual'][0])
    
    # Define colors for different scenarios in the order they are defined
    colors = [COLOR_0, COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5]
    
    # remove -extra from scenario names
    scenario_names = [name.replace('-extra', '') for name in scenario_names]

    # Create bars for each scenario
    bars = plt.bar(scenario_names, heatwave_values, color=colors[:len(scenario_names)], alpha=0.7, label='Heatwave')
    plt.bar(scenario_names, annual_values, color=colors[:len(scenario_names)], alpha=0.5, label='Annual')
    
    # Add percentage labels on top of each bar
    for i, (bar, percentage, heatwave_val, annual_val) in enumerate(zip(bars, percentages, heatwave_values, annual_values)):
        # Add percentage label
        plt.text(bar.get_x() + bar.get_width()/2, heatwave_val-2, f'{percentage:.1f}%', 
                ha='center', va='top', fontsize=11, color='white', fontweight='bold')
        # Add heatwave value label
        plt.text(bar.get_x() + bar.get_width()/2, heatwave_val, f'{heatwave_val:.0f}', 
                ha='center', va='bottom', fontsize=9, color='black')
        plt.text(bar.get_x() + bar.get_width()/2, annual_val, f'{annual_val:.0f}', 
                ha='center', va='bottom', fontsize=9, color='black')

    # Add labels and title
    if key == 'cooling':
        plt.ylabel('Cooling Energy Demand (kWh)')
        plt.title(f'{key.capitalize()} Energy Demand Heatwave Contribution (2020)')
    elif key == 'energy':
        plt.ylabel('Energy Use (kWh)')
        plt.title(f'{key.capitalize()} Energy Use Heatwave Contribution (2020)')
    elif key == 'cost':
        plt.ylabel('Cooling Energy Cost (€)')
        plt.title(f'{key.capitalize()} Cooling Energy Cost Heatwave Contribution (2020)')
    elif key == 'overheating':
        plt.ylabel('Overheated Hours (h)')
        plt.title(f'{key.capitalize()} Heatwave Contribution (2020)')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    plt.grid(True, axis='y', alpha=0.3, linestyle=':')
        # Add legend to bottom left corner
    plt.legend(loc='lower left')
    
    # Adjust layout to prevent label cutoff
    # plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    # Define all scenarios available in the heatwave file
    scenario_name_0 = "baseline"
    scenario_name_1 = "S01_simple_shading-extra"
    scenario_name_2 = "S02_simple_paint"
    scenario_name_3 = "S04_simple_combi"
    scenario_name_4 = "M01_medium_shading"
    scenario_name_5 = "D01_deep_shading-paint"

    # concat scenario names - include all available scenarios
    scenario_names = [scenario_name_0, scenario_name_1, scenario_name_2, scenario_name_3, 
                     scenario_name_4, scenario_name_5]

    key = 'overheating'
    # Load baseline scenario
    with open(os.path.join(current_dir, 'scenarios', 'baseline_scenario.json')) as f:
        baseline_scenario = json.load(f)['baseline']

    # Load renovation scenarios
    with open(os.path.join(current_dir, 'scenarios', 'renovation_scenarios.json')) as f:
        renovation_scenarios = json.load(f)
    
    # Load universal data
    with open(os.path.join(current_dir, 'config', 'universal_data.json')) as f:
        universal_data = json.load(f)


    heatwave_year = [2020]
    # load scenario_comfort.csv
    scenario_comfort_csv = os.path.join(current_dir, 'scenarios', 'scenario_comfort.csv')


    # Combine baseline and renovation scenarios into a single configuration dictionary
    all_scenario_configs = {'baseline': baseline_scenario}
    all_scenario_configs.update(renovation_scenarios)
    
    # Create comparison for multiple scenarios using the new function
    print("\n" + "="*50)
    print("MULTIPLE SCENARIO COMPARISON")
    print("="*50)
    scenario_results = compare_heatwave_impact_multiple_scenarios(scenario_names, all_scenario_configs, years=heatwave_year, key=key)
    plot_heatwave_comparison_multiple_scenarios(scenario_results, key=key)



"""
heatwave_years = [1995, 1997, 1999, 2003, 2006, 2013, 2018, 2019, 2020, 2022]
top6_years = [1995, 2006, 2018, 2019, 2020, 2022]
 
# Create comparison DataFrame and plot for single scenario (original function)
df_comparison = compare_heatwave_impact(all_values_base_csv, heat_wave_file_2020, years=heatwave_years, cop=cop_0, key='cooling')
plot_heatwave_comparison(df_comparison, key='cooling')
"""