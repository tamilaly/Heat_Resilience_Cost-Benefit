import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy import stats
from numpy.polynomial.polynomial import Polynomial

# This script can be used to generate samples for the savings distribution
# The samples are then saved as a fixed set

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Scenario to simulate
scenario_name = "D01_deep_shading"
key = "cooling"
num_pdf_samples = 10000
output_file = os.path.join(current_dir, 'scenarios', 'samples', f'{scenario_name}_{key}_savings.csv')

# Load universal data and analysis settings
with open(os.path.join(current_dir, 'config', 'universal_data.json')) as f:
    universal_data = json.load(f)

with open(os.path.join(current_dir, 'config', 'analysis_settings.json')) as f:
    analysis_settings = json.load(f)

# Load baseline scenario
with open(os.path.join(current_dir, 'scenarios', 'baseline_scenario.json')) as f:
    baseline_data = json.load(f)['baseline']

# Load renovation scenarios
with open(os.path.join(current_dir, 'scenarios', 'renovation_scenarios.json')) as f:
    renovation_scenarios = json.load(f)

scenario  = renovation_scenarios[scenario_name]
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
def generate_combined_kde_samples(baseline_data, scenario_data, universal_data, variable_name, num_samples=1000, percentiles=[0, 5, 25, 50, 75, 95, 100]):
    """
    Generate KDE samples for savingsfrom cooling demand data.
    
    Args:
        baseline_data (dict): Dictionary containing baseline scenario information including cooling_file and cooling_system_type
        scenario_data (dict): Dictionary containing scenario information including cooling_file and cooling_system_type
        universal_data (dict): Dictionary containing universal parameters including cooling COP values
        variable_name (str): Name of the variable to generate samples from
        num_samples (int): Number of samples to generate
        percentiles (list): List of percentiles to use for interpolation
        
    Returns:
        numpy.ndarray: Array of KDE samples
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

    if variable_name == "heating":
        # Add fixed costs for the heating source
        b_fixed_cost_key = f"fixed_cost_{baseline_data['heating_energy_source']}"
        b_fixed_cost = universal_data[b_fixed_cost_key]
        b_total_fixed_cost = b_fixed_cost * baseline_data['household_size']

        s_fixed_cost_key = f"fixed_cost_{scenario_data['heating_energy_source']}"
        s_fixed_cost = universal_data[s_fixed_cost_key]
        s_total_fixed_cost = s_fixed_cost * scenario_data['household_size']

        # Get heating energy price
        b_heating_energy_source = baseline_data['heating_energy_source']
        b_price_per_kwh = universal_data[f"price_{b_heating_energy_source}"]

        s_heating_energy_source = scenario_data['heating_energy_source']
        s_price_per_kwh = universal_data[f"price_{s_heating_energy_source}"]

        # Calculate the fixed cost difference
        fixed_cost_difference = b_total_fixed_cost - s_total_fixed_cost

        # calculate heating cost
        b_cost = baseline_energy_use * b_price_per_kwh
        s_cost = scenario_energy_use * s_price_per_kwh

    else:
        # Get heating energy price
        b_cooling_energy_source = baseline_data['cooling_energy_source']
        b_price_per_kwh = universal_data[f"price_{b_cooling_energy_source}"]

        s_cooling_energy_source = scenario_data['cooling_energy_source']
        s_price_per_kwh = universal_data[f"price_{s_cooling_energy_source}"]

        # calculate heating cost
        b_cost = baseline_energy_use * b_price_per_kwh
        s_cost = scenario_energy_use * s_price_per_kwh
    
    # Convert percentiles to probabilities
    probabilities = np.array(percentiles) / 100.0

    # Create an interpolation function
    interp_func_baseline = interp1d(probabilities, b_cost, bounds_error=False, fill_value="extrapolate")
    interp_func_scenario = interp1d(probabilities, s_cost, bounds_error=False, fill_value="extrapolate")

    # Generate uniform random samples
    uniform_samples = np.random.uniform(0, 1, 5000)
    #  
    samples_baseline = interp_func_baseline(uniform_samples)
    samples_scenario = interp_func_scenario(uniform_samples)
    
    # Fit KDE and generate samples
    kde_baseline = gaussian_kde(samples_baseline, bw_method=0.6)

    # reconstruct two kde curves for both scenarios
    kde_scenario = gaussian_kde(samples_scenario, bw_method=0.6)

    # cut off the min and max value bounds
    # Apply bounds
    min_bound_baseline = min(b_cost)
    max_bound_baseline = max(b_cost)
    min_bound_scenario = min(s_cost)
    max_bound_scenario = max(s_cost)

    # generate a number of percentile references
    up_bound = 99
    low_bound = 1
    steps = 1000
    step_size = (up_bound - low_bound) / steps
    percentile_set = np.arange(low_bound, up_bound, step_size)

    x_vals_base = np.linspace(min_bound_baseline, max_bound_baseline, 1000)
    x_vals_scenario = np.linspace(min_bound_scenario, max_bound_scenario, 1000)

    # find percentiles with numpy
    baseline_percentiles = find_percentile_vals(kde_baseline, x_vals_base, percentile_set)
    scenario_percentiles = find_percentile_vals(kde_scenario, x_vals_scenario, percentile_set)

    # add min and max bound to baseline and scenario percentiles
    baseline_percentiles = np.insert(baseline_percentiles, 0, min_bound_baseline)
    baseline_percentiles = np.append(baseline_percentiles, max_bound_baseline)
    scenario_percentiles = np.insert(scenario_percentiles, 0, min_bound_scenario)
    scenario_percentiles = np.append(scenario_percentiles, max_bound_scenario)

    # add 0 and 100 to percentiles
    percentiles = np.insert(percentile_set, 0, 0)
    percentiles = np.append(percentiles, 100)

    if variable_name == "heating":
        # calculate the savings
        savings = baseline_percentiles - scenario_percentiles + fixed_cost_difference
    else:
        savings = baseline_percentiles - scenario_percentiles

    min_savings = min(savings)
    max_savings = max(savings)

    # Create KDE for savings
    kde_savings = gaussian_kde(savings, bw_method=0.6)
    x_vals_savings = np.linspace(min(savings), max(savings), 1000)
    pdf_kde_savings = kde_savings(x_vals_savings)

    # generate samples
    savings_samples = kde_savings.resample(num_samples)[0]

    # delete samples out of bounds if min bound <0 
    if min_savings < 0:
        savings_samples = savings_samples[(savings_samples >= 0)]
    else:
        savings_samples = savings_samples[(savings_samples >= min_savings)]
    
    # plot the savings samples
    plt.hist(savings_samples, bins=50, density=True, alpha=0.3, color='purple', label='savings samples')
    plt.show()


    """
    # Fit normal distribution
    mu, std = stats.norm.fit(savings)
    norm_dist = stats.norm(mu, std)
    # x_range = np.linspace(min_bound_savings, max_bound_savings, num_samples)
    # norm_pdf = stats.norm.pdf(x_range, mu, std)  # Get PDF

    # Generate samples from normal distribution
    norm_samples = norm_dist.rvs(num_samples)

    # delete samples out of bounds
    norm_samples = norm_samples[(norm_samples >= min_bound_savings)] # & (norm_samples <= max_bound_savings)
    """
    # for Cooling returns savings as kWh and for Heating returns savings as cost
    return kde_savings, savings_samples

if key == "cooling":
    # Make a combined savings cost dataframe for COOLING
    kde, samples = generate_combined_kde_samples(baseline_data, scenario, universal_data, variable_name="cooling", num_samples=num_pdf_samples, percentiles=[0, 3, 27, 46, 76, 97, 100])
   
elif key == "heating":
    # Make a combined savings cost dataframe for HEATING
    kde, samples = generate_combined_kde_samples(baseline_data, scenario, universal_data, variable_name="heating", num_samples=num_pdf_samples, percentiles=[0, 11, 25, 51, 75, 96, 100])

# make a dataframe with the samples
if key == "cooling":
    samples_df = pd.DataFrame(samples, columns=['cooling_savings_samples'])
elif key == "heating":
    samples_df = pd.DataFrame(samples, columns=['heating_savings_samples'])

# save the dataframe to a csv file
samples_df.to_csv(output_file, index=False)
print(f"Samples saved to {output_file}")