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

# create a folder for the plots
plots_dir = os.path.join(current_dir, 'results', 'Cooling', 'EUI')
os.makedirs(plots_dir, exist_ok=True)

# Define colors - RGB values normalized to 0-1 range
r, g, b = 112/255, 44/255, 168/255
# r, g, b = 5/255, 0/255, 113/255

COLOR_MAIN = (r, g, b)
COLOR_HIST = tuple(0.5 * (c + 1) for c in COLOR_MAIN)
COLOR_RED = (204/255, 31/255, 4/255)
COLOR_BLUE = (5/255, 0/255, 113/255)
COLOR_GREEN = (98/255, 177/255, 163/255)
COLOR_PINK = (237/255, 145/255, 185/255)

COLOR_0 = (242/255, 99/255, 102/255)
COLOR_1 = (243/255, 182/255, 68/255)
COLOR_1_LIGHT = tuple(0.5 * (c + 1) for c in COLOR_1)
COLOR_2 = 'darkorange'
COLOR_3 = (98/255, 177/255, 163/255)
COLOR_4 = (237/255, 145/255, 185/255)
COLOR_5 = (5/255, 0/255, 113/255)

HIGHLIGHT_1 = (COLOR_RED)
HIGHLIGHT_2 = (237/255, 145/255, 185/255)
HIGHLIGHT_3 = (277/255, 144/255, 132/255)
def find_percentiles(kde, x_vals, values):
    
    # KDE estimation
    pdf_kde = kde(x_vals)
    cdf_kde = np.cumsum(pdf_kde)
    cdf_kde = cdf_kde / cdf_kde[-1]
    
    # Interpolation function: CDF(x) → x
    cdf = interp1d(x_vals, cdf_kde)

    percentiles = []
    for v in values:
        # Get x-values for given percentiles
        percentiles.append(float(cdf(v)))

    return percentiles

def find_percentile_vals(kde, x_vals, values):
    
    # KDE estimation
    pdf_kde = kde(x_vals)
    cdf_kde = np.cumsum(pdf_kde)
    cdf_kde = cdf_kde / cdf_kde[-1]
    
    # Interpolation function: CDF(x) → x
    inv_cdf = interp1d(cdf_kde, x_vals)

    percentile_values = []
    for v in values:
        # Get x-values for given percentiles
        percentile_values.append(float(inv_cdf(v/100)))  # Convert percentile to probability (0-1)

    return np.array(percentile_values)  # Convert list to numpy array

def reconstruct_pdf_from_percentiles(scenario_data,
                                     universal_data,
                                     scenario_name,
                                     output_column='Cooling_Annual_Total',
                                     percentiles=[0, 5, 25, 50, 75, 95, 100], 
                                     num_samples=1000, 
                                     plot=True,
                                     key='cost',
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
    # extract first word of output_column and make lowercase
    e_type = output_column.split('_')[0].lower()
    # remove cooling or heating from outputcolumn
    indicator = output_column.replace(f'{e_type}_', '')

    # open cooling or heating file
    csv_path = os.path.join(current_dir, 'scenarios', scenario_data[f'{e_type}_file'])

    # Energy system efficiency
    cop = scenario_data[f'{e_type}_cop_or_eer']

    # Read the CSV file
    df = pd.read_csv(csv_path)
    
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

    # get the price per kwh
    price_per_kwh = universal_data[f"price_{scenario_data[f'{e_type}_energy_source']}"]
    fixed_cost = universal_data[f"fixed_cost_{scenario_data[f'{e_type}_energy_source']}"]

    if key == 'cost':
        sorted_values = sorted_values * price_per_kwh

    # Convert percentiles to probabilities (0-1 range)
    probabilities = np.array(percentiles) / 100.0
    
    marker_max = max(sorted_values)

    # Method 1: Interpolation between percentiles
    # Create an interpolation function
    interp_func = interp1d(probabilities, sorted_values, bounds_error=False, fill_value="extrapolate")

    # Generate uniform random samples
    uniform_samples = np.random.uniform(0, 1, num_samples)
    
    # Transform uniform samples using the inverse CDF (interpolation function)
    samples = interp_func(uniform_samples)

    """
    # use uniform samples to fit a normal distribution
    mu, std = stats.norm.fit(samples)
    print(mu, std)
    x_range = np.linspace(min(samples), max(samples), 1000)
    norm_pdf = stats.norm.pdf(x_range, mu, std)  # Get PDF"""
    
    # This is an alternative approach that might give smoother results
    kde = stats.gaussian_kde(samples, bw_method=0.6)
    kde_samples = kde.resample(num_samples)[0]

    
    # min and max bound
    min_bound = min(sorted_values)

    kde_samples = kde_samples[kde_samples >= min_bound]
    # kde_samples = kde_samples[kde_samples <= max_bound]
    x_vals = np.linspace(min(sorted_values), max(sorted_values), 500)
    marker_85 = find_percentile_vals(kde, x_vals, [85])[0]
    
    # np.percentile(kde_samples, 85)

    # Calculate and print statistics
    mean_pdf = np.mean(kde_samples)

    # Calculate the cost
    if key == 'heating':
        marker_85_cost = marker_85 * price_per_kwh + fixed_cost
        mean_pdf_cost = mean_pdf * price_per_kwh + fixed_cost
        marker_max_cost = marker_max * price_per_kwh + fixed_cost
    else:
        marker_85_cost = marker_85 * price_per_kwh
        mean_pdf_cost = mean_pdf * price_per_kwh
        marker_max_cost = marker_max * price_per_kwh
        
    print(f"\nDistribution Statistics: {output_column} Energy Use for {scenario_name}")
    print(f"Mean PDF: {mean_pdf:.2f} => {mean_pdf_cost:.2f} €")
    print(f"85th percentile: {marker_85:.2f} => {marker_85_cost:.2f} €")
    print(f"2020 maximum recorded: {marker_max:.2f} => {marker_max_cost:.2f} €")

    if plot:
                
        plt.figure(figsize=(12, 6))
        # Plot the original data points
        plt.scatter(sorted_values, probabilities, color=color_main, s=50, label='year values')
        # label years on the plot
        for i, (value, year) in enumerate(zip(sorted_values, sorted_years)):
            plt.text(value*1.05, probabilities[i]*0.97, str(year), color=color_main, fontsize=10)
        
        # Plot the interpolation function
        x_interp = np.linspace(0, 1, 1000)
        y_interp = interp_func(x_interp)
        plt.plot(y_interp, x_interp, color_main, alpha=0.5, label='Interpolation')
        plt.xlabel("Cooling Energy Use Intensity (kWh per m² per year)")
        plt.ylabel('Cumulative Probability (%)')
        plt.title(f'{scenario_name} CDF Energy Use')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'{scenario_name}_CDF_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        scenario_name = scenario_name.replace('_', ' ').replace('-extra', '').capitalize()
        # Plot the KDE curve
        plt.figure(figsize=(12, 6))
        # Plot histograms of the generated samples
        plt.hist(kde_samples, bins=50, density=True, alpha=0.3, color=color_hist, label='KDE Samples')
        # Plot the KDE curve
        x_kde = np.linspace(min(kde_samples), max(kde_samples), 1000)
        plt.plot(x_kde, kde(x_kde), color_main, label='KDE')

        # mark the mean
        plt.scatter(mean_pdf, kde(mean_pdf), color=color_main, s=100)
        plt.axvline(x=mean_pdf, color=color_main, linestyle=':', label=f'Mean: {mean_pdf:.2f} kWh/m²')
        # mark the 85th percentile
        plt.scatter(marker_85, kde(marker_85), color=HIGHLIGHT_2, s=100)
        plt.axvline(x=marker_85, color=HIGHLIGHT_2, linestyle=':', label=f'85th percentile: {marker_85:.2f} kWh/m²')
        # mark the max
        plt.scatter(marker_max, kde(marker_max), color=HIGHLIGHT_1, s=100)
        plt.axvline(x=marker_max, color=HIGHLIGHT_1, linestyle=':', label=f'2020 maximum recorded: {marker_max:.2f} kWh/m²')
        if key == 'cost':
            plt.xlabel(f"{e_type.capitalize()} Energy Cost (€ per year)")
        else:
            plt.xlabel(f"{e_type.capitalize()} Energy Use (kWh/m² per year)")
        plt.ylabel('Probability')
        plt.title(f'{scenario_name}: Reconstructed Probability Density Function')
        plt.legend()
        plt.grid(True)
        
        # Save the plot before showing it
        if indicator == 'EUI':
            plt.savefig(os.path.join(plots_dir, f'{scenario_name}_EUI_distribution.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(plots_dir, f'{scenario_name}_{indicator}_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()

    return samples, kde_samples, marker_85, marker_max

def reconstruct_pdf_overlay(scenario_name, baseline_data, scenario_data, universal_data, variable_name, percentiles=[0, 5, 25, 50, 75, 95, 100], plot=True, savings=True, num_samples=1000):
    """
    Reconstruct a probability density function from a limited set of percentiles.
    
    Parameters:
    -----------
    csv_file_baseline : str
        Path to the CSV file containing the baseline data
    csv_file_scenario : str
        Path to the CSV file containing the scenario data
    scenario_name : str
        Name of the scenario
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
    
    file_key = f"{variable_name}_file"
    # Read values from cooling demand file
    baseline_cooling_file = os.path.join(current_dir, 'scenarios', baseline_data[file_key])
    baseline_cooling_df = pd.read_csv(baseline_cooling_file)
    scenario_cooling_file = os.path.join(current_dir, 'scenarios', scenario_data[file_key])
    scenario_cooling_df = pd.read_csv(scenario_cooling_file)
    
    # Capitalize the variable name
    key = f"{variable_name.capitalize()}_Annual_Total"
    # Sort values
    sorted_baseline_df = baseline_cooling_df.sort_values(by=key)
    sorted_baseline_values = sorted_baseline_df[key].values
    sorted_scenario_df = scenario_cooling_df.sort_values(by=key)
    sorted_scenario_values = sorted_scenario_df[key].values
    
    # Get the cooling COP from the scenario
    if variable_name == "cooling":
        baseline_cop = universal_data[f"{baseline_data['cooling_system_type']}_cop"]
        scenario_cop = universal_data[f"{scenario_data['cooling_system_type']}_cop"]
    elif variable_name == "heating":
        baseline_cop = universal_data[f"{baseline_data['heating_system_type']}_eer"]
        scenario_cop = universal_data[f"{scenario_data['heating_system_type']}_eer"]
    
    # Calculate cooling energy use
    baseline_energy_use = sorted_baseline_values / baseline_cop
    scenario_energy_use = sorted_scenario_values / scenario_cop

    # find energy use for 2020
    # baseline_energy_use_2020 = baseline_energy_use[sorted_baseline_df['Year'] == 2020][0]
    # scenario_energy_use_2020 = scenario_energy_use[sorted_scenario_df['Year'] == 2020][0]

    # Convert percentiles to probabilities
    probabilities = np.array(percentiles) / 100.0

    # Create an interpolation function
    interp_func_baseline = interp1d(probabilities, baseline_energy_use, bounds_error=False, fill_value="extrapolate")
    interp_func_scenario = interp1d(probabilities, scenario_energy_use, bounds_error=False, fill_value="extrapolate")

    # Generate uniform random samples
    uniform_samples = np.random.uniform(0, 1, 5000)
    samples_baseline = interp_func_baseline(uniform_samples)
    samples_scenario = interp_func_scenario(uniform_samples)
    
    # Fit KDE and generate samples
    kde_baseline = gaussian_kde(samples_baseline, bw_method=0.6)
    kde_samples_baseline = kde_baseline.resample(num_samples)[0]

    # reconstruct two kde curves for both scenarios
    kde_scenario = gaussian_kde(samples_scenario, bw_method=0.6)
    kde_samples_scenario = kde_scenario.resample(num_samples)[0]

    # cut off the min and max value bounds
    # Apply bounds
    min_bound_baseline = min(baseline_energy_use)
    max_bound_baseline = max(baseline_energy_use)
    min_bound_scenario = min(scenario_energy_use)
    max_bound_scenario = max(scenario_energy_use)

    kde_samples_baseline = kde_samples_baseline[kde_samples_baseline > min_bound_baseline]
    kde_samples_baseline = kde_samples_baseline[kde_samples_baseline < max_bound_baseline]
    kde_samples_scenario = kde_samples_scenario[kde_samples_scenario > min_bound_scenario]
    kde_samples_scenario = kde_samples_scenario[kde_samples_scenario < max_bound_scenario]
    
    # generate a number of percentile references
    up_bound = 99
    low_bound = 1
    steps = 1000
    step_size = (up_bound - low_bound) / steps
    percentiles = np.arange(low_bound, up_bound, step_size)

    x_vals_base = np.linspace(min_bound_baseline, max_bound_baseline, 1000)
    x_vals_scenario = np.linspace(min_bound_scenario, max_bound_scenario, 1000)

    baseline_percentiles = find_percentile_vals(kde_baseline, x_vals_base, percentiles)
    scenario_percentiles = find_percentile_vals(kde_scenario, x_vals_scenario, percentiles)

    # add min and max bound to baseline and scenario percentiles
    baseline_percentiles = np.insert(baseline_percentiles, 0, min_bound_baseline)
    baseline_percentiles = np.append(baseline_percentiles, max_bound_baseline)
    scenario_percentiles = np.insert(scenario_percentiles, 0, min_bound_scenario)
    scenario_percentiles = np.append(scenario_percentiles, max_bound_scenario)
    # add 0 and 100 to percentiles
    percentiles = np.insert(percentiles, 0, 0)
    percentiles = np.append(percentiles, 100)

    # threshhold percentile 85%
    p_limit = [15, 85]
    p_limit_2 = [65]
    threshold_percentile_base = find_percentile_vals(kde_baseline, x_vals_base, p_limit)
    threshold_percentile_base_2 = find_percentile_vals(kde_baseline, x_vals_base, p_limit_2)[0]
    threshold_percentile_scenario = find_percentile_vals(kde_scenario, x_vals_scenario, p_limit)
    threshold_percentile_scenario_2 = find_percentile_vals(kde_scenario, x_vals_scenario, p_limit_2)[0]
    price = 0.243  # €/kWh

    print(f"Threshold percentile base 2: {round(threshold_percentile_base_2, 2)} kWh")
    print(f"Threshold percentile scenario 2: {round(threshold_percentile_scenario_2, 2)} kWh")
    savings_p_65 = round(threshold_percentile_base_2 - threshold_percentile_scenario_2, 2)
    print(f"Savings p_65: {round(savings_p_65, 2)} kWh => {round(savings_p_65 * price, 2)} €")
     
    print(f"{scenario_name} scenario 15th percentile: {round(threshold_percentile_scenario[0], 2)} kWh => {round(threshold_percentile_scenario[0] * price, 2)} €,\n85th percentile: {round(threshold_percentile_scenario[1], 2)} kWh => {round(threshold_percentile_scenario[1] * price, 2)} €")
    
    savings_threshold_down = round(threshold_percentile_base[0] - threshold_percentile_scenario[0], 2)
    savings_threshold_up = round(threshold_percentile_base[1] - threshold_percentile_scenario[1], 2)
    print(f"Savings threshold: {round(savings_threshold_down, 2)} kWh => {round(savings_threshold_down * price, 2)} €,\n{round(savings_threshold_up, 2)} kWh => {round(savings_threshold_up * price, 2)} €")

    # construct combined dataframe for baseline and scenario percentiles
    combined_df = pd.DataFrame({'Percentile': percentiles, 'Baseline': baseline_percentiles, 'Scenario': scenario_percentiles})
    
    if plot:
        # plot both kde curves
        plt.figure(figsize=(12, 8))
        # Plot the KDE curve
        x_kde_baseline = np.linspace(min(kde_samples_baseline), max(kde_samples_baseline), 1000)
        plt.plot(x_kde_baseline, kde_baseline(x_kde_baseline), color= COLOR_0, label='pdf baseline')
        x_kde_scenario = np.linspace(min(kde_samples_scenario), max(kde_samples_scenario), 1000)
        plt.plot(x_kde_scenario, kde_scenario(x_kde_scenario), color= COLOR_1, label='pdf scenario')

        if scenario_name == 'S01_simple_shading-extra':
            color_scenario = COLOR_1 
        elif scenario_name == 'S02_simple_paint':
            color_scenario = COLOR_2
        elif scenario_name == 'S04_simple_combi':
            color_scenario = COLOR_3    
        else:
            color_scenario = COLOR_0

        # mark the percentiles on the plot
        plt.scatter(baseline_percentiles, kde_baseline(baseline_percentiles), color=COLOR_0)
        plt.scatter(scenario_percentiles, kde_scenario(scenario_percentiles), color=color_scenario)
        
        # Plot histograms of the generated samples
        plt.hist(kde_samples_baseline, bins=50, density=True, alpha=0.3, color= COLOR_0, label='samples baseline')
        plt.hist(kde_samples_scenario, bins=50, density=True, alpha=0.3, color= color_scenario, label='samples scenario')

        # Add vertical lines for Threshold percentiles
        plt.axvline(x=threshold_percentile_base[1], color=COLOR_0, linestyle=':', label='85th percentile base')
        plt.scatter(threshold_percentile_base[1], kde_baseline(threshold_percentile_base[1]), color=COLOR_0, s=35)
        plt.axvline(x=threshold_percentile_scenario[1], color=color_scenario, linestyle=':', label='85th percentile scenario')
        plt.scatter(threshold_percentile_scenario[1], kde_scenario(threshold_percentile_scenario[1]), color=color_scenario, s=35)

        plt.xlabel("Cooling Energy Use (kWh per year)")
        plt.ylabel('Probability Density')
        plt.title(f'Reconstructed Probability Density Functions for baseline and scenario {scenario_name}')
        plt.legend()
        # plt.show()
    
    if savings:
        # Calculate average baseline to scenario values
        average_bs = (baseline_percentiles + scenario_percentiles) / 2

        # 85th percentile savings
        percentile_85_baseline = threshold_percentile_base[1]
        percentile_85_scenario = threshold_percentile_scenario[1]
        savings_85th_percentile = percentile_85_baseline - percentile_85_scenario
        average_85th_percentile = (percentile_85_baseline + percentile_85_scenario) / 2
        average_65th_percentile = (threshold_percentile_base_2 + threshold_percentile_scenario_2) / 2

        # Energy prices 
        cooling_energy_source = baseline_data[f'{variable_name}_energy_source']
        price_per_kwh = universal_data[f"price_{cooling_energy_source}"]

        marker_savings = savings_85th_percentile
        marker = average_85th_percentile
        savings, marker_threshold, min_savings_average_bs = construct_savings_distribution(baseline_percentiles, scenario_percentiles, marker, marker_savings, price_per_kwh=price_per_kwh, num_samples=num_samples, key='energy')

        # add savings to combined dataframe
        combined_df['Savings'] = savings

        min_savings_above_85th_percentile = marker_threshold
        print(f"min_savings_above_marker: {round(marker_threshold, 2)} => {round(marker_threshold * price_per_kwh, 2)} €")

        # find marker_threshold percentile in combined        
        # Find the second closest matching percentile for marker_threshold
        differences = np.abs(combined_df['Savings'] - marker_threshold)
        # Get indices of sorted differences
        sorted_indices = np.argsort(differences)
        # Get second closest index
        second_closest_idx = sorted_indices[1]
        marker_threshold_percentile = combined_df.iloc[second_closest_idx]['Percentile']
        print(f"Second closest corresponding percentile: {round(marker_threshold_percentile, 2)} %")

        # maximum of threshold savings
        next_closest_idx = sorted_indices[2]
        marker_max_savings = combined_df.iloc[next_closest_idx]['Savings']
        marker_max_average = (combined_df.iloc[next_closest_idx]['Baseline']+combined_df.iloc[next_closest_idx]['Scenario'])/2

        if plot:
            # linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(average_bs, savings)
            # print(f"slope: {round(slope, 3)}, intercept: {round(intercept, 3)}, r_value: {round(r_value, 3)}, p_value: {round(p_value, 3)}, std_err: {round(std_err, 3)}")

            line = slope * average_bs + intercept
            point_at_line = (savings_p_65 - intercept)/slope

            plt.figure(figsize=(14, 8))
            plt.scatter(average_bs, savings, color=COLOR_1_LIGHT, label='savings at percentiles')
            plt.scatter(average_65th_percentile, savings_p_65, color=COLOR_RED, s=35, label='65th percentile')
            plt.axhline(y=savings_p_65, color=COLOR_RED, alpha=0.5, linestyle=':')

            plt.scatter(average_85th_percentile, savings_85th_percentile, color=COLOR_GREEN, s=35, label='85th percentile')
            plt.axvline(x=average_85th_percentile, color=COLOR_GREEN, alpha=0.5, linestyle=':')

            plt.scatter(min_savings_average_bs, min_savings_above_85th_percentile, color=COLOR_BLUE, s=35, label='min savings above 85th percentile')
            
            plt.plot(average_bs, line, color=COLOR_PINK, alpha=0.7, label='linear regression')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.xlabel('Demand Average Baseline to Scenario in kWh per year')
            plt.ylabel('Energy Savings in kWh per year')
            plt.title(f'S01: Scatterplot of Energy Demand vs Savings')
            plt.legend()
            plt.show()

    # return both kdes
    return kde_baseline, kde_scenario, kde_samples_baseline, kde_samples_scenario

def construct_savings_distribution(baseline_percentiles, scenario_percentiles, marker, marker_savings, num_samples=1000, price_per_kwh=0.1, key='energy'):
    
    # calculate the savings
    savings = baseline_percentiles - scenario_percentiles

    # Calculate average baseline to scenario values
    average_bs = (baseline_percentiles + scenario_percentiles) / 2

    savings_above_marker = savings[average_bs >= marker]
    average_bs_above_marker = average_bs[average_bs >= marker]

    # find min savings above marker
    min_savings_above_marker = min(savings_above_marker)
    marker_threshold = round(min_savings_above_marker, 2)

    # find the min savings index in the savings list
    min_savings_above_marker_index = np.argmin(savings_above_marker)
    min_savings_average_bs =average_bs_above_marker[min_savings_above_marker_index]
    
    if key == 'cost':
        savings = savings * price_per_kwh
        marker_threshold = marker_threshold * price_per_kwh

    # mu, std = stats.norm.fit(savings)
    #print(f"mu: {round(mu, 3)}, std: {round(std, 3)}")
    # x_range = np.linspace(min(savings), max(savings), 1000)
    #norm_pdf = stats.norm.pdf(x_range, mu, std)  # Get PDF
    #norm_dist = stats.norm(mu, std)

    # Generate samples from normal distribution
    # norm_samples = norm_dist.rvs(num_samples)

    # Create KDE for savings
    kde_savings = gaussian_kde(savings, bw_method=0.6)
    kde_samples_savings = kde_savings.resample(10000)[0]
    kde_samples_savings = kde_samples_savings[kde_samples_savings >= min(savings)]
    kde_samples_savings = kde_samples_savings[kde_samples_savings <= max(savings)]
    x_vals_savings = np.linspace(min(savings), max(savings), 1000)
    pdf_kde_savings = kde_savings(x_vals_savings)

    # plot the savings
    plt.figure(figsize=(14, 8))
    plt.hist(kde_samples_savings, bins=50, density=True, color=COLOR_1_LIGHT, label='savings samples')

    # plot the marker with axv line
    plt.axvline(x=marker_threshold, color=COLOR_RED, linestyle=':')
    
    # Shade the area under the curve from marker to max using KDE
    mask = x_vals_savings >= marker_threshold
    plt.fill_between(x_vals_savings[mask], 0, pdf_kde_savings[mask], color=COLOR_RED, alpha=0.3, label=f'Savings threshold 65th percentile: {round(marker_threshold, 2)} kWh')

    # plt.plot(x_range, norm_pdf, color=COLOR_MAIN, linestyle='--', label='normal distribution')
    plt.plot(x_vals_savings, pdf_kde_savings, color=COLOR_1, label='kde distribution')
    if key == 'energy':
        plt.xlabel('Energy Savings (kWh per year)')
    elif key == 'cost':
        plt.xlabel('Cost Savings (€ per year)')
    plt.ylabel('Probability Density')
    plt.title('Probability Density Function of Energy Savings')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return savings, marker_threshold, min_savings_average_bs

def reconstruct_combined_pdf_savings(scenario_name, baseline_data, scenario_data, universal_data, variable_name, percentiles=[0, 5, 25, 50, 75, 95, 100], plot=False, num_samples=1000, color_main='purple', color_hist='mediumpurple'):

    """
    Reconstruct a probability density function from a limited set of percentiles.
    
    Parameters:
    -----------
    csv_file_baseline : str
        Path to the CSV file containing the baseline data
    csv_file_scenario : str
        Path to the CSV file containing the scenario data
    scenario_name : str
        Name of the scenario
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
    file_key = f"{variable_name}_file"
    # Read values from cooling demand file
    baseline_cooling_file = os.path.join(current_dir, 'scenarios', baseline_data[file_key])
    baseline_cooling_df = pd.read_csv(baseline_cooling_file)
    scenario_cooling_file = os.path.join(current_dir, 'scenarios', scenario_data[file_key])
    scenario_cooling_df = pd.read_csv(scenario_cooling_file)
    
    # Capitalize the variable name
    key = f"{variable_name.capitalize()}_Annual_Total"
    # Sort values
    sorted_baseline_df = baseline_cooling_df.sort_values(by=key)
    sorted_baseline_values = sorted_baseline_df[key].values
    sorted_scenario_df = scenario_cooling_df.sort_values(by=key)
    sorted_scenario_values = sorted_scenario_df[key].values
    
    print(sorted_baseline_df)
    print(sorted_scenario_df)

    # Get the cooling COP from the scenario
    if variable_name == "cooling":
        baseline_cop = universal_data[f"{baseline_data['cooling_system_type']}_cop"]
        scenario_cop = universal_data[f"{scenario_data['cooling_system_type']}_cop"]
    elif variable_name == "heating":
        baseline_cop = universal_data[f"{baseline_data['heating_system_type']}_eer"]
        scenario_cop = universal_data[f"{scenario_data['heating_system_type']}_eer"]
    
    # Calculate cooling energy use
    baseline_energy_use = sorted_baseline_values / baseline_cop
    scenario_energy_use = sorted_scenario_values / scenario_cop
    
    # Calculate savings
    savings = baseline_energy_use - scenario_energy_use

    # Convert percentiles to probabilities (0-1 range)
    probabilities = np.array(percentiles) / 100.0

    # Method 1: Interpolation between percentiles
    # Create an interpolation function
    interp_func = interp1d(probabilities, savings, bounds_error=False, fill_value="extrapolate")

    # Generate uniform random samples
    uniform_samples = np.linspace(0, 1, 1000) # np.random.uniform(0, 1, 1000)
    samples = interp_func(uniform_samples)
    
    # Fit KDE and generate samples
    kde = gaussian_kde(samples, bw_method=0.6)
    kde_samples = kde.resample(num_samples)[0]
    # remove negative values
    kde_samples = kde_samples[kde_samples >= 0]

    # Sort df by the output column in ascending order
    sorted_years = sorted_baseline_df['Year'].values
    # Construct Dataframe for sorted years and savings
    sorted_df = pd.DataFrame({'Year': sorted_years, 'Savings': savings})
    
    # print year and savings
    print(sorted_df)

    if plot:

        # Plot the consumption and savings
        plt.figure(figsize=(12, 8))
        x = np.arange(len(sorted_years))
        width = 0.35  # Width of the bars
        # Create the bars side by side
        bars_1 = plt.bar(x-width/2, sorted_baseline_df[key], width, color=color_main, label='reference years')
        bars_2 = plt.bar(x+width/2, sorted_scenario_df[key], width, color=color_hist, label='scenario years')	
        

        # Create a second y-axis for savings
        ax2 = plt.gca().twinx()
        # Create line plot for savings on the second y-axis
        ax2.scatter(x, sorted_df['Savings'], marker='o', color='tab:red', s=50, label='savings')
        ax2.plot(x, sorted_df['Savings'], color='tab:red', linestyle='-', linewidth=1, label='_nolegend_')  # Remove label from line plot
        # ax2.set_ylabel('Energy Savings (kWh per year)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(bottom=0, top=500)  # Set maximum value to 500
        
        # Add labels and title
        # Set the x-axis ticks and labels
        plt.xticks(x, sorted_years)
        ax1 = plt.gca()
        ax1.yaxis.set_label_position('left')
        ax1.set_ylabel("Cooling Energy Use (kWh per year)")
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel('Energy Savings (kWh per year)', color='black')

        # Make a string from variable with spaces between words
        scenario_name_str = ' '.join(word for word in scenario_name.replace('-extra','').split('_'))
        plt.title(f'{scenario_name_str} Cooling Energy Use and Savings')

        # Combine legends from both axes
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Remove duplicate labels
        unique_labels = []
        unique_lines = []
        for line, label in zip(lines1 + lines2, labels1 + labels2):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_lines.append(line)
        plt.gca().legend(unique_lines, unique_labels, loc='upper right')
        # plt.grid(True)
        plt.show()

        # Plot CDF of savings
        plt.figure(figsize=(12, 8))
        # Plot the original data points
        plt.scatter(savings, probabilities, color=color_main, s=10, label='year values')
        
        # label years on the plot
        value_2020 = sorted_df[sorted_df['Year'] == 2020]['Savings'].values[0]
        prob_2020 = probabilities[sorted_years == 2020][0]
        plt.text(value_2020*1.01, prob_2020*0.98, '2020', color='tab:red', fontsize=10)
        plt.scatter(value_2020, prob_2020, color='tab:red', s=25)

        # Plot the interpolation function
        x_interp = np.linspace(0, 1, 1000)
        y_interp = interp_func(x_interp)
        plt.plot(y_interp, x_interp, color_main, alpha=0.5, label='Interpolation')
        plt.xlabel("Cooling Energy Savings (kWh per year)")
        plt.ylabel('Cumulative Probability (%)')
        plt.title(f'{scenario_name} CDF Energy Savings')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the KDE curve
        plt.figure(figsize=(12, 8))
        # Plot histograms of the generated samples
        plt.hist(kde_samples, bins=50, density=True, alpha=0.3, color=color_hist, label='KDE Samples')
        
        # Add vertical lines for 2020 and 2007 values
        value_2020 = sorted_df[sorted_df['Year'] == 2020]['Savings'].values[0]
        value_2007 = sorted_df[sorted_df['Year'] == 2007]['Savings'].values[0]
        plt.axvline(x=value_2020, color='tab:red', linestyle=':', label='min & max values')
        plt.scatter(value_2020, kde(value_2020), color='tab:red', s=35, label='2020')
        plt.axvline(x=value_2007, color='tab:red', linestyle=':')

        # Plot the KDE curve
        x_kde = np.linspace(min(kde_samples), max(kde_samples), 1000)
        plt.plot(x_kde, kde(x_kde), color_main, label='KDE')

        plt.xlabel("Cooling Energy Savings (kWh per year)")
        plt.ylabel('Probability')
        plt.title(f'{scenario_name}: Reconstructed Probability Density Function')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return samples, kde_samples
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
def boxplot_comparison(scenario_names, s0_samples_cooling, s1_samples_cooling, s2_samples_cooling, s3_samples_cooling, s4_samples_cooling, s5_samples_cooling, markers_85th=None, markers_max=None):
    # Create a figure for the boxplot
    plt.figure(figsize=(12, 8))
    
    # Create a list of samples for each scenario
    samples_list = [s0_samples_cooling, s1_samples_cooling, s2_samples_cooling, s3_samples_cooling, s4_samples_cooling, s5_samples_cooling]
    
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

    # Create x positions for the means
    x_positions = list(range(1, len(means) + 1))

    # Plot means as scatter points
    plt.scatter(x_positions, means, color='black', s=100, zorder=3, label='Mean')
    # Overlay the mean as a thick black line
    for i, mean in enumerate(means, start=1):
        plt.plot([i-0.25, i+0.25], [mean, mean], color='black', linewidth=3, label='_nolegend_')

    # Add markers for 85th percentile and max
    if markers_85th:
        for i, marker in enumerate(markers_85th, start=1):
            plt.scatter(i, marker, color=HIGHLIGHT_2, s=100, label='85th percentile' if i == 1 else "")
            plt.plot([i-0.25, i+0.25], [marker, marker], color=HIGHLIGHT_2, linewidth=1, label='_nolegend_')

    if markers_max:
        for i, marker in enumerate(markers_max, start=1):
            plt.scatter(i, marker, color=HIGHLIGHT_1, s=100, label='2020 max recorded' if i == 1 else "")
            plt.plot([i-0.25, i+0.25], [marker, marker], color=HIGHLIGHT_1, linewidth=1, label='_nolegend_')

    # Add labels and title
    plt.ylabel('Energy Use Intensity (kWh/m²per year)')
    # fix y-axis to 0-20
    # plt.ylim(0, 18)
    plt.title('Heating Energy Use Distribution Comparison Across Scenarios')
    # Rotate x-axis labels if they're long
    plt.xticks(rotation=45, ha='right')
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    # put the legend in the top right corner
    plt.legend(loc='upper right')
    plt.subplots_adjust(bottom=0.2)  # Increase the bottom margin
    
    # Save the plot before showing it
    plt.savefig(os.path.join(plots_dir, 'scenario_comparison_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # make a second just line plot with the mean and the 85th percentile and max 2020
    plt.figure(figsize=(12, 8))
    # Create x positions for the means
    x_positions = list(range(1, len(means) + 1))
    plt.plot(x_positions, means, color='black', linewidth=1, label='_nolegend_')
    plt.scatter(x_positions, means, color='black', s=50, zorder=3, label='Mean')
    plt.plot(x_positions, markers_85th, color=HIGHLIGHT_2, linewidth=1)
    plt.scatter(x_positions, markers_85th, color=HIGHLIGHT_2, s=50, zorder=3, label='85th percentile')
    plt.plot(x_positions, markers_max, color=HIGHLIGHT_1, linewidth=1)
    plt.scatter(x_positions, markers_max, color=HIGHLIGHT_1, s=50, zorder=3, label='2020 max recorded')
    plt.ylabel('Energy Use Intensity (kWh/m²per year)')
    plt.title('Cooling Energy Use Distribution Comparison Across Scenarios')
    plt.legend()
    plt.show()
def run_cooling_eui_distribution(baseline_0, scenario_name_1, scenario_name_2, scenario_name_3, scenario_name_4, scenario_name_5, plot=False):
    
    variable_name = "Cooling_EUI"
    probabilities = [0, 3, 27, 46, 76, 97, 100]
    color_main = "navy"
    color_hist = "lightsteelblue"
    # concat scenario names
    scenario_names = [baseline_0, scenario_name_1, scenario_name_2, scenario_name_3, scenario_name_4, scenario_name_5]

    # floor_area = 505.33
    # Load baseline scenario
    with open(os.path.join(current_dir, 'scenarios', 'baseline_scenario.json')) as f:
        baseline_scenario = json.load(f)['baseline']

    # Load renovation scenarios
    with open(os.path.join(current_dir, 'scenarios', 'renovation_scenarios.json')) as f:
        renovation_scenarios = json.load(f)
    
    with open(os.path.join(current_dir, 'config', 'universal_data.json')) as f:
        universal_data = json.load(f)

    num_samples = 50000

    key = 'cooling'

    # Load Scenario 0
    scenario_0  = baseline_scenario

    # SAMPLES
    s0_samples_cooling, s0_kde_samples_cooling, marker_85_0, marker_max_0 = reconstruct_pdf_from_percentiles(baseline_scenario,
                                                            scenario_name=baseline_0,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,  
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples,
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)
    # Load Scenario 1
    scenario_1  = renovation_scenarios[scenario_name_1]

    # SAMPLES
    s1_samples_cooling, s1_kde_samples_cooling, marker_85_1, marker_max_1 = reconstruct_pdf_from_percentiles(scenario_1,
                                                            scenario_name=scenario_name_1,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples,
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)

    # Load Scenario 2
    scenario_2  = renovation_scenarios[scenario_name_2]

    # SAMPLES
    s2_samples_cooling, s2_kde_samples_cooling, marker_85_2, marker_max_2 = reconstruct_pdf_from_percentiles(scenario_2,
                                                            scenario_name=scenario_name_2,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples, 
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)

    # Load Scenario 3
    scenario_3  = renovation_scenarios[scenario_name_3]
    # SAMPLES
    s3_samples_cooling, s3_kde_samples_cooling, marker_85_3, marker_max_3 = reconstruct_pdf_from_percentiles(scenario_3,
                                                            scenario_name=scenario_name_3,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples, 
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)
    
    # Load Scenario 4
    scenario_4  = renovation_scenarios[scenario_name_4]
    # SAMPLES
    s4_samples_cooling, s4_kde_samples_cooling, marker_85_4, marker_max_4 = reconstruct_pdf_from_percentiles(scenario_4,
                                                            scenario_name=scenario_name_4,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples, 
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)
    
    # Load Scenario 5
    scenario_5  = renovation_scenarios[scenario_name_5]
    # SAMPLES
    s5_samples_cooling, s5_kde_samples_cooling, marker_85_5, marker_max_5 = reconstruct_pdf_from_percentiles(scenario_5,
                                                            scenario_name=scenario_name_5,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples, 
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)

    markers_85th = [marker_85_0, marker_85_1, marker_85_2, marker_85_3, marker_85_4, marker_85_5]
    markers_max = [marker_max_0, marker_max_1, marker_max_2, marker_max_3, marker_max_4, marker_max_5]

    # Boxplot comparison
    boxplot_comparison(scenario_names, s0_kde_samples_cooling, s1_kde_samples_cooling, s2_kde_samples_cooling, s3_kde_samples_cooling, s4_kde_samples_cooling, s5_kde_samples_cooling, markers_85th, markers_max)
def run_heating_eui_distribution(baseline_0, scenario_name_1, scenario_name_2, scenario_name_3, scenario_name_4, scenario_name_5, plot=False):
    
    variable_name = "Heating_EUI"
    probabilities = [0, 11, 25, 51, 75, 96, 100]
    color_main = "tab:red"
    color_hist = "lightcoral"
    # concat scenario names
    scenario_names = [baseline_0, scenario_name_1, scenario_name_2, scenario_name_3, scenario_name_4, scenario_name_5]

    # floor_area = 505.33
    # Load baseline scenario
    with open(os.path.join(current_dir, 'scenarios', 'baseline_scenario.json')) as f:
        baseline_scenario = json.load(f)['baseline']

    # Load renovation scenarios
    with open(os.path.join(current_dir, 'scenarios', 'renovation_scenarios.json')) as f:
        renovation_scenarios = json.load(f)
    
    with open(os.path.join(current_dir, 'config', 'universal_data.json')) as f:
        universal_data = json.load(f)

    num_samples = 50000

    key = 'heating'

    # Load Scenario 0
    scenario_0  = baseline_scenario

    # SAMPLES
    s0_samples_heating, s0_kde_samples_heating, marker_85_0, marker_max_0 = reconstruct_pdf_from_percentiles(baseline_scenario,
                                                            scenario_name=baseline_0,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,  
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples,
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)

    # Load Scenario 1
    scenario_1  = renovation_scenarios[scenario_name_1]

    # SAMPLES
    s1_samples_heating, s1_kde_samples_heating, marker_85_1, marker_max_1 = reconstruct_pdf_from_percentiles(scenario_1,
                                                            scenario_name=scenario_name_1,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples,
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)

    # Load Scenario 2
    scenario_2  = renovation_scenarios[scenario_name_2]

    # SAMPLES
    s2_samples_heating, s2_kde_samples_heating, marker_85_2, marker_max_2 = reconstruct_pdf_from_percentiles(scenario_2,
                                                            scenario_name=scenario_name_2,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples, 
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)

    # Load Scenario 3
    scenario_3  = renovation_scenarios[scenario_name_3]
    # SAMPLES
    s3_samples_heating, s3_kde_samples_heating, marker_85_3, marker_max_3 = reconstruct_pdf_from_percentiles(scenario_3,
                                                            scenario_name=scenario_name_3,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples, 
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)
    
    # Load Scenario 4
    scenario_4  = renovation_scenarios[scenario_name_4]
    # SAMPLES
    s4_samples_heating, s4_kde_samples_heating, marker_85_4, marker_max_4 = reconstruct_pdf_from_percentiles(scenario_4,
                                                            scenario_name=scenario_name_4,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples, 
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)
    
    # Load Scenario 5
    scenario_5  = renovation_scenarios[scenario_name_5]
    # SAMPLES
    s5_samples_heating, s5_kde_samples_heating, marker_85_5, marker_max_5 = reconstruct_pdf_from_percentiles(scenario_5,
                                                            scenario_name=scenario_name_5,
                                                            universal_data=universal_data,
                                                            output_column=variable_name,
                                                            percentiles=probabilities, 
                                                            num_samples=num_samples, 
                                                            key=key,
                                                            plot=plot,
                                                            color_main=color_main,
                                                            color_hist=color_hist)

    # Boxplot comparison
    boxplot_comparison(scenario_names, s0_kde_samples_heating, s1_kde_samples_heating, s2_kde_samples_heating, s3_kde_samples_heating, s4_kde_samples_heating, s5_kde_samples_heating)

if __name__ == "__main__":
    scenario_name_0 = "baseline"
    scenario_name_1 = "S01_simple_shading-extra"
    scenario_name_2 = "S02_simple_paint"
    scenario_name_3 = "S04_simple_combi"
    scenario_name_4 = "M01_medium_shading"
    scenario_name_5 = "D01_deep_shading-paint"
    
    variable_name = "Cooling_EUI" # "Heating_Annual_Total"
    key = 'cooling'
    probabilities = [0, 3, 27, 46, 76, 97, 100] # [0, 11, 25, 51, 75, 96, 100] 
    color_main = "navy"
    color_hist = "lightsteelblue"
    num_samples = 50000
    # concat scenario names
    scenario_names = [scenario_name_0, scenario_name_1, scenario_name_2, scenario_name_3]

    # floor_area = 505.33
    # Load baseline scenario
    with open(os.path.join(current_dir, 'scenarios', 'baseline_scenario.json')) as f:
        baseline_scenario = json.load(f)['baseline']

    # Load renovation scenarios
    with open(os.path.join(current_dir, 'scenarios', 'renovation_scenarios.json')) as f:
        renovation_scenarios = json.load(f)
    
    # Load universal data
    with open(os.path.join(current_dir, 'config', 'universal_data.json')) as f:
        universal_data = json.load(f)


    scenario = baseline_scenario

    run_cooling_eui_distribution(scenario_name_0, scenario_name_1, scenario_name_2, scenario_name_3, scenario_name_4, scenario_name_5, plot=False)

    # call function reconstruct_pdf_from_percentiles
    reconstruct_pdf_from_percentiles(scenario, universal_data, scenario_name_0, output_column=variable_name, percentiles=probabilities, num_samples=num_samples, plot=True, key='energy', color_main=color_main, color_hist=color_hist)

    # reconstruct_pdf_overlay(scenario_name_3, baseline_scenario, scenario, universal_data, key, percentiles=probabilities, plot=False, num_samples=25000, savings=True)
    # kde_0, kde_1, kde_samples_0, kde_samples_1 = reconstruct_pdf_overlay(scenario_name_1, baseline_scenario, scenario, universal_data, key, percentiles=probabilities, plot=False, savings=True, num_samples=num_samples)

    # mean of the energy use
    # mean_energy_use_baseline = np.mean(kde_samples_0)
    # mean_energy_use_scenario = np.mean(kde_samples_1)
    # print(f"Mean energy use for {scenario_name_0}: {mean_energy_use_baseline} kWh")
    # print(f"Mean energy use for {scenario_name_1}: {mean_energy_use_scenario} kWh")

    # I need the mean , 85th percentile and the max of the energy use; then plot those three points for each scenario incomparison to the baseline

