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
r, g, b = 204/255, 31/255, 4/255
COLOR_MAIN = (r, g, b)
COLOR_HIST = tuple(0.5 * (c + 1) for c in COLOR_MAIN)
COLOR_0 = 242/255, 99/255, 102/255
COLOR_1 = (243/255, 182/255, 68/255)
COLOR_2 = 'darkorange'
COLOR_3 = (98/255, 177/255, 163/255)
COLOR_4 = (237/255, 145/255, 185/255)
COLOR_5 = (5/255, 0/255, 113/255)

HIGHLIGHT_1 = (204/255, 31/255, 4/255)
HIGHLIGHT_2 = (237/255, 145/255, 185/255)
HIGHLIGHT_3 = (277/255, 144/255, 132/255)

def reconstruct_cdf_from_percentiles(csv_file,
                                     scenario_name,
                                     output_column='Cooling_Annual_Total',
                                     cop=None,
                                     percentiles=[0, 5, 25, 50, 75, 95, 100], 
                                     ):
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
   
    return sorted_values, sorted_years, interp_func

def stats_from_pdf(scenario_name, samples):

    mean_pdf = np.mean(samples)
    std_pdf = np.std(samples)

    print(f"\nDistribution Statistics: {scenario_name} Carbon Intensity")
    # Mean
    print(f"Mean PDF: {mean_pdf:.2f}")
    # Standard Deviation
    print(f"Standard Deviation: {std_pdf:.2f}")
    # COnfidence Intervall
    print(f"95% Confidence Interval: [{mean_pdf - 1.96*std_pdf:.2f}, {mean_pdf + 1.96*std_pdf:.2f}]")
    # print min and max
    print(f"Min: {np.min(samples):.2f}; Max: {np.max(samples):.2f}")
def plot_pdf(kde_samples, kde, scenario_name, output_column, color_main='purple', color_hist='mediumpurple'):

    # Plot the KDE curve
    plt.figure(figsize=(12, 8))
    # Plot histograms of the generated samples
    plt.hist(kde_samples, bins=50, density=True, alpha=0.3, color=color_hist, label='KDE Samples')
    
    # Plot the KDE curve
    x_kde = np.linspace(min(kde_samples), max(kde_samples), 1000)
    plt.plot(x_kde, kde(x_kde), color_main, label='KDE')

    plt.xlabel(output_column + ' [kgCO2/m2]')
    plt.ylabel('Probability')
    plt.title(f'{scenario_name}: Reconstructed Probability Density Function')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cdf(sorted_values, probabilities, sorted_years, scenario_name, output_column, color_main='purple'):
    
    plt.figure(figsize=(12, 8))
    # Plot the original data points
    plt.scatter(sorted_values, probabilities, color=color_main, s=50, label='year values')
    # label years on the plot
    for i, (value, year) in enumerate(zip(sorted_values, sorted_years)):
        plt.text(value+10, probabilities[i]-0.05, str(year), color=color_main, fontsize=10)
    
    # Plot the interpolation function
    x_interp = np.linspace(0, 1, 1000)
    y_interp = interp_func(x_interp)
    plt.plot(y_interp, x_interp, color_main, alpha=0.5, label='Interpolation')
    plt.xlabel(output_column)
    plt.ylabel('Cumulative Probability (%)')
    plt.title(f'{scenario_name} CDF Energy Use')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_carbon_intensity(values, scenario, universal_data, key='heating', area=100):
    # get energy source from scenario
    if key == 'heating':    
        energy_source = scenario['heating_energy_source']
    elif key == 'cooling':
        energy_source = scenario['cooling_energy_source']
        
    key_carbon_intensity = f"CE_rate_{energy_source}"
    # get carbon intensity from universal data
    carbon_intensity = universal_data[key_carbon_intensity]
    # calculate carbon intensity
    carbon_intensity = (values * carbon_intensity) / area

    return carbon_intensity

def analyze_carbon_intensity(scenario, scenario_name, universal_data, key='heating', area=100, num_samples=50000):
    """
    Perform full carbon intensity analysis for a given scenario.
    
    Parameters:
    -----------
    scenario : dict
        Scenario dictionary containing energy source and file information
    scenario_name : str
        Name of the scenario for output labeling
    universal_data : dict
        Dictionary containing carbon emission rates
    key : str
        'heating' or 'cooling' to indicate which energy type to analyze
    area : float
        Floor area for normalization
    num_samples : int
        Number of samples to generate for the analysis
        
    Returns:
    --------
    kde_samples : numpy.ndarray
        KDE samples of the carbon intensity distribution
    """
    # Get file path and efficiency based on key
    if key == 'heating':
        csv_path = os.path.join(current_dir, 'scenarios', scenario['heating_file'])
        cop = scenario['heating_cop_or_eer']
        output_column = 'Heating_Annual_Total'
        percentiles = [0, 11, 25, 51, 75, 96, 100]
    else:  # cooling
        csv_path = os.path.join(current_dir, 'scenarios', scenario['cooling_file'])
        cop = scenario['cooling_cop_or_eer']
        output_column = 'Cooling_Annual_Total'
        percentiles = [0, 3, 27, 46, 76, 97, 100]

    # Reconstruct CDF and get interpolation function
    sorted_values, sorted_years, interp_func = reconstruct_cdf_from_percentiles(
        csv_path,
        scenario_name=scenario_name,
        output_column=output_column,
        cop=cop,
        percentiles=percentiles,
    )


    # Generate samples
    uniform_samples = np.random.uniform(0, 1, 5000)
    samples = interp_func(uniform_samples)

    # Calculate carbon intensity
    carbon_intensity_samples = calculate_carbon_intensity(
        samples, 
        scenario, 
        universal_data, 
        key=key, 
        area=area
    )
    
    # Generate KDE samples
    kde = stats.gaussian_kde(carbon_intensity_samples, bw_method=0.6)
    kde_samples = kde.resample(num_samples)[0]
    
    # Calculate min_value from carbon intensity samples instead of energy values
    min_value = min(carbon_intensity_samples)
    # get min and max of the sorted values
    marker_max = max(carbon_intensity_samples)
    kde_samples = kde_samples[kde_samples >= min_value]
    
    # Safety check to ensure kde_samples is not empty
    if len(kde_samples) == 0:
        print(f"Warning: All KDE samples were filtered out for {scenario_name}. Using original carbon intensity samples.")
        kde_samples = carbon_intensity_samples

    marker_85 = np.percentile(kde_samples, 85)

    # Print statistics
    print(f"Analysis for {scenario_name} ({key})")
    stats_from_pdf(scenario_name, carbon_intensity_samples)

    return kde_samples, marker_85, marker_max

def boxplot_comparison(scenario_names, key, samples_list, markers_85th=None, markers_max=None):
    # Create a figure for the boxplot
    plt.figure(figsize=(14, 8))
    
    # remove '-extra' from scenario names
    scenario_names = [name.replace('-extra', '') for name in scenario_names]

    # Create a boxplot
    boxplot = plt.boxplot(samples_list, tick_labels=scenario_names, patch_artist=True, showfliers=False)
    
    # Customize boxplot colors
    colors = [COLOR_0, COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5]
    for patch, color in zip(boxplot['boxes'], colors[:len(samples_list)]):
        patch.set_facecolor(color)
    
    # Calculate means for each scenario
    means = [np.mean(samples) for samples in samples_list]

    # Overlay the mean as a thick black line
    for i, mean in enumerate(means, start=1):
        plt.plot([i-0.25, i+0.25], [mean, mean], color='black', linewidth=3, label='_nolegend_')
        # Plot means as scatter points
        plt.scatter(i, mean, color='black', s=100, zorder=3, label='Mean' if i == 1 else '_nolegend_')
    
    # Add markers for 85th percentile and max
    if markers_85th:
        for i, marker in enumerate(markers_85th, start=1):
            plt.scatter(i, marker, color=HIGHLIGHT_2, s=100, label='85th percentile' if i == 1 else "")
            plt.plot([i-0.25, i+0.25], [marker, marker], color=HIGHLIGHT_2, linewidth=1, label='_nolegend_')

    if markers_max:
        for i, marker in enumerate(markers_max, start=1):
            plt.scatter(i, marker, color=HIGHLIGHT_1, s=100, label='2020 max recorded' if i == 1 else "")
            plt.plot([i-0.25, i+0.25], [marker, marker], color=HIGHLIGHT_1, linewidth=1, label='2020 max recorded' if i == 1 else '_nolegend_')

    # Add labels and title
    plt.ylabel('Carbon Intensity (kgCO2/m²per year)')
    plt.title(f'Carbon Intensity Distribution of {key} Comparison Across Scenarios')
    # Rotate x-axis labels if they're long
    plt.xticks(rotation=45, ha='right')
    # Adjust layout to prevent label cutoff
    plt.ylim(0, 7)
    plt.tight_layout()
    plt.legend(loc='upper right')

    plt.subplots_adjust(bottom=0.2)  # Increase the bottom margin
    plt.savefig(os.path.join(current_dir, 'results', 'CEI', f'CEI_boxplot_comparison_{key}.png'), dpi=300)
    plt.show()

def small_monte_carlo(kde_samples_1, kde_samples_2, output_path, num_samples=1000):

    """
    Perform a small Monte Carlo simulation to combine the two distributions
    the two values are added together
    """
    combined_samples = []
    for i in range(num_samples):
        # sample from the first distribution
        sample_1 = np.random.choice(kde_samples_1)
        # sample from the second distribution
        sample_2 = np.random.choice(kde_samples_2)
        # add the two values together
        combined_samples.append(round(sample_1 + sample_2, 2))
        # print(f"Combined sample {i}: {combined_samples[i]}")

    output_column_name = f"Carbon_Intensity_Combined"

    # save the combined samples to a csv file
    df = pd.DataFrame({output_column_name: combined_samples})
    df.index.name = 'Index'
    
    df.to_csv(output_path)

    return combined_samples

def analyze_heating_carbon_intensity(baseline_scenario, renovation_scenarios, universal_data, scenario_names, floor_area, num_samples):
    """
    Analyze heating carbon intensity for all scenarios.
    
    Parameters:
    -----------
    baseline_scenario : dict
        Baseline scenario configuration
    renovation_scenarios : dict
        Dictionary of renovation scenarios
    universal_data : dict
        Universal data containing carbon emission rates
    scenario_names : list
        List of scenario names to analyze
    floor_area : float
        Floor area for normalization
    num_samples : int
        Number of samples to generate
        
    Returns:
    --------
    tuple : (kde_samples_list, markers_85th, markers_max)
        kde_samples_list: List of KDE samples for each scenario
        markers_85th: List of 85th percentile markers
        markers_max: List of maximum markers
    """
    kde_samples_list = []
    markers_85th = []
    markers_max = []
    
    # Analyze baseline scenario
    kde_samples_baseline, marker_85_0, marker_max_0 = analyze_carbon_intensity(
        baseline_scenario,
        scenario_names[0],
        universal_data,
        key='heating',
        area=floor_area,
        num_samples=num_samples
    )
    kde_samples_list.append(kde_samples_baseline)
    markers_85th.append(marker_85_0)
    markers_max.append(marker_max_0)
    
    # Analyze renovation scenarios
    for i, scenario_name in enumerate(scenario_names[1:], 1):
        scenario = renovation_scenarios[scenario_name]
        kde_samples, marker_85, marker_max = analyze_carbon_intensity(
            scenario,
            scenario_name,
            universal_data,
            key='heating',
            area=floor_area,
            num_samples=num_samples
        )
        kde_samples_list.append(kde_samples)
        markers_85th.append(marker_85)
        markers_max.append(marker_max)
    
    return kde_samples_list, markers_85th, markers_max

def analyze_cooling_carbon_intensity(baseline_scenario, renovation_scenarios, universal_data, scenario_names, floor_area, num_samples):
    """
    Analyze cooling carbon intensity for all scenarios.
    
    Parameters:
    -----------
    baseline_scenario : dict
        Baseline scenario configuration
    renovation_scenarios : dict
        Dictionary of renovation scenarios
    universal_data : dict
        Universal data containing carbon emission rates
    scenario_names : list
        List of scenario names to analyze
    floor_area : float
        Floor area for normalization
    num_samples : int
        Number of samples to generate
        
    Returns:
    --------
    tuple : (kde_samples_list, markers_85th, markers_max)
        kde_samples_list: List of KDE samples for each scenario
        markers_85th: List of 85th percentile markers
        markers_max: List of maximum markers
    """
    kde_samples_list = []
    markers_85th = []
    markers_max = []
    
    # Analyze baseline scenario
    kde_samples_baseline, marker_85_0, marker_max_0 = analyze_carbon_intensity(
        baseline_scenario,
        scenario_names[0],
        universal_data,
        key='cooling',
        area=floor_area,
        num_samples=num_samples
    )
    kde_samples_list.append(kde_samples_baseline)
    markers_85th.append(marker_85_0)
    markers_max.append(marker_max_0)
    
    # Analyze renovation scenarios
    for i, scenario_name in enumerate(scenario_names[1:], 1):
        scenario = renovation_scenarios[scenario_name]
        kde_samples, marker_85, marker_max = analyze_carbon_intensity(
            scenario,
            scenario_name,
            universal_data,
            key='cooling',
            area=floor_area,
            num_samples=num_samples
        )
        kde_samples_list.append(kde_samples)
        markers_85th.append(marker_85)
        markers_max.append(marker_max)
    
    return kde_samples_list, markers_85th, markers_max

def save_carbon_intensity_results(samples_list, scenario_names, key, output_dir):
    """
    Save carbon intensity results to CSV files.
    
    Parameters:
    -----------
    samples_list : list
        List of KDE samples for each scenario
    scenario_names : list
        List of scenario names
    key : str
        'heating' or 'cooling'
    output_dir : str
        Output directory for saving files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual scenario results
    for i, (samples, scenario_name) in enumerate(zip(samples_list, scenario_names)):
        # Clean scenario name for filename
        clean_name = scenario_name.replace('-', '_').replace(' ', '_')
        filename = f"carbon_intensity_{key}_{clean_name}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df = pd.DataFrame({
            'Carbon_Intensity_kgCO2_m2_year': samples
        })
        df.index.name = 'Sample_Index'
        df.to_csv(filepath, index=True)
        print(f"Saved {filename}")
    
    # Save combined results
    combined_filename = f"carbon_intensity_{key}_all_scenarios.csv"
    combined_filepath = os.path.join(output_dir, combined_filename)
    
    combined_data = {}
    for samples, scenario_name in zip(samples_list, scenario_names):
        clean_name = scenario_name.replace('-', '_').replace(' ', '_')
        combined_data[f'Carbon_Intensity_{clean_name}'] = samples
    
    # Pad shorter arrays with NaN to make them the same length
    max_length = max(len(samples) for samples in samples_list)
    for key_name in combined_data:
        if len(combined_data[key_name]) < max_length:
            combined_data[key_name] = np.append(combined_data[key_name], 
                                              [np.nan] * (max_length - len(combined_data[key_name])))
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.index.name = 'Sample_Index'
    combined_df.to_csv(combined_filepath, index=True)
    print(f"Saved {combined_filename}")

if __name__ == "__main__":
    scenario_name_0 = "baseline"
    scenario_name_1 = "S01_simple_shading-extra"
    scenario_name_2 = "S02_simple_paint"
    scenario_name_3 = "S04_simple_combi"
    scenario_name_4 = "M01_medium_shading"
    scenario_name_5 = "D01_deep_shading-paint"
    key_1 = 'heating'
    key_2 = 'cooling'
    # concat scenario names
    scenario_names = [scenario_name_0, scenario_name_1, scenario_name_2, scenario_name_3, scenario_name_4, scenario_name_5]

    floor_area = 505.33
    num_samples = 50000

    # Load baseline scenario
    with open(os.path.join(current_dir, 'scenarios', 'baseline_scenario.json')) as f:
        baseline_scenario = json.load(f)['baseline']

    # Load renovation scenarios
    with open(os.path.join(current_dir, 'scenarios', 'renovation_scenarios.json')) as f:
        renovation_scenarios = json.load(f)

    # load universal data
    with open(os.path.join(current_dir, 'config', 'universal_data.json')) as f:
        universal_data = json.load(f)
    
    # Analyze heating carbon intensity for all scenarios
    print("=== HEATING CARBON INTENSITY ANALYSIS ===")
    heating_samples, heating_markers_85th, heating_markers_max = analyze_heating_carbon_intensity(
        baseline_scenario,
        renovation_scenarios,
        universal_data,
        scenario_names,
        floor_area,
        num_samples
    )
    
    # Analyze cooling carbon intensity for all scenarios
    print("\n=== COOLING CARBON INTENSITY ANALYSIS ===")
    cooling_samples, cooling_markers_85th, cooling_markers_max = analyze_cooling_carbon_intensity(
        baseline_scenario,
        renovation_scenarios,
        universal_data,
        scenario_names,
        floor_area,
        num_samples
    )

    # Boxplot comparison cooling
    boxplot_comparison(scenario_names, 
                       'Cooling',
                       cooling_samples,
                       markers_85th=cooling_markers_85th,
                       markers_max=cooling_markers_max)