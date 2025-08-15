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


# The purpose this script is to calculate the NPV distribution for a given scenario
# The NPV is the sum of the discounted cash flows
# The script will calculate the NPV for a given scenario and using the baseline cost as benefit cost
# the calculated NPVs are saved in a csv file including the corresponding energy savings samples
# additionally for every run of the Monte Carlo simulation the results are added to a general statistics csv file

# Setting for the Study by Lalyko
# bw method 0.6 to smooth distribution
# interest rate 3.0% 

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Scenario to simulate
scenario_name = "S04_simple_combi"
cooling = True # True or False
heating = True # True or False

# Number of simulations for Monte Carlo
num_simulations = 1000 # num of simulations
num_pdf_samples = 10000 # num of samples for the probability density function

# Toggles for saving the output
save_output = True # True or False
calculate_statistics = True # True or False

# Load the output file
folder = pd.Timestamp.now().strftime('%y-%m-%d')
accumulated_npv = 0 # not used

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

def calculate_maintenance_cost(scenario, start_year, analysis_period):
    maintenance_costs = pd.Series(0.0, index=range(start_year, start_year + analysis_period))
    
    for key in scenario:
        if key.startswith("maintenance_cost_item"):
            item_num = key.replace("maintenance_cost_item", "")
            cost = scenario[key]
            cycle_key = f"maintenance_cycle_item{item_num}"
            if cycle_key in scenario:
                cycle = scenario[cycle_key]
                if cycle is not None and cycle > 0:
                    for year in range(start_year + cycle, start_year + analysis_period, cycle):
                        maintenance_costs[year] += cost
    
    return maintenance_costs
def calculate_replacement_cost(scenario, start_year, analysis_period):
    replacement_costs = pd.Series(0.0, index=range(start_year, start_year + analysis_period))
    
    for key in scenario:
        if key.startswith("replacement_cost_item"):
            item_num = key.replace("replacement_cost_item", "")
            cost = scenario[key]
            cycle_key = f"replacement_cycle_item{item_num}"
            if cycle_key in scenario:
                cycle = scenario[cycle_key]
                if cycle is not None and cycle > 0:
                    for year in range(start_year + cycle, start_year + analysis_period, cycle):
                        replacement_costs[year] += cost
    
    return replacement_costs

def apply_discount(df, discount_map):
    discounted = df.copy()
    for year in discounted.index:
        discounted.loc[year] *= discount_map[year]
    return discounted

# Source the analysis settings
start_year = analysis_settings["start_year"]
reference_year = analysis_settings["reference_year"]
analysis_period = analysis_settings["analysis_period"]
years = list(range(start_year, start_year + analysis_period))

# Set up the dataframe
df = pd.DataFrame(0.0, index=years, columns=["Investment Cost Difference", "Operational Savings", "Maintenance Difference", "Replacement Difference"])

# only for fixed sample sets
cooling_samples_file = os.path.join(current_dir, 'scenarios', 'samples', f'{scenario_name}_cooling_savings.csv')
heating_samples_file = os.path.join(current_dir, 'scenarios', 'samples', f'{scenario_name}_heating_savings.csv')

if cooling == True and heating == False:
    # Calculate the investment cost difference
    df.loc[start_year, "Investment Cost Difference"] = (baseline_data["investment_cost_cooling"]) - (scenario["investment_cost_cooling"])
    # Make a combined savings cost dataframe for COOLING
    cooling_samples_df = pd.read_csv(cooling_samples_file)
    c_combi_scenario_samples = cooling_samples_df['cooling_savings_samples'].values
    # c_pdf, c_combi_scenario_samples = generate_combined_kde_samples(baseline_data, scenario, universal_data, variable_name="cooling", num_samples=num_pdf_samples, percentiles=[0, 3, 27, 46, 76, 97, 100])
  
elif cooling == False and heating == True:
    # Calculate the investment cost difference
    df.loc[start_year, "Investment Cost Difference"] = (baseline_data["investment_cost_heating"]) - (scenario["investment_cost_heating"])
    # Make a combined savings cost dataframe for HEATING
    heating_samples_df = pd.read_csv(heating_samples_file)
    h_combi_scenario_samples = heating_samples_df['heating_savings_samples'].values
    # h_kde, h_combi_scenario_samples = generate_combined_kde_samples(baseline_data, scenario, universal_data, variable_name="heating", num_samples=num_pdf_samples, percentiles=[0, 11, 25, 51, 75, 96, 100])

else:
    # Calculate the investment cost difference
    df.loc[start_year, "Investment Cost Difference"] = (baseline_data["investment_cost_cooling"]+baseline_data["investment_cost_heating"]) - (scenario["investment_cost_cooling"]+scenario["investment_cost_heating"])
    # Make a combined savings cost dataframe for COOLING
    cooling_samples_df = pd.read_csv(cooling_samples_file)
    c_combi_scenario_samples = cooling_samples_df['cooling_savings_samples'].values
    # c_pdf, c_combi_scenario_samples = generate_combined_kde_samples(baseline_data, scenario, universal_data, variable_name="cooling", num_samples=num_pdf_samples, percentiles=[0, 3, 27, 46, 76, 97, 100])
    # Make a combined savings cost dataframe for HEATING
    heating_samples_df = pd.read_csv(heating_samples_file)
    h_combi_scenario_samples = heating_samples_df['heating_savings_samples'].values
    # h_pdf, h_combi_scenario_samples = generate_combined_kde_samples(baseline_data, scenario, universal_data, variable_name="heating", num_samples=num_pdf_samples, percentiles=[0, 11, 25, 51, 75, 96, 100])

# calculate maintenance and replacement costs
baseline_maintenance_costs = calculate_maintenance_cost(baseline_data, start_year, analysis_period) # baseline maintenance costs
baseline_replacement_costs = calculate_replacement_cost(baseline_data, start_year, analysis_period) # baseline replacement costs
scenario_maintenance_costs = calculate_maintenance_cost(scenario, start_year, analysis_period) # scenario maintenance costs
scenario_replacement_costs = calculate_replacement_cost(scenario, start_year, analysis_period) # scenario replacement costs


# Monte Carlo Simulation
npv_results = []

for i in range(num_simulations):

    # Create a fresh DataFrame for each simulation
    sim_df = pd.DataFrame(0.0, index=years, columns=["Investment Cost Difference", "Operational Savings", "Maintenance Difference", "Replacement Difference"])

    # Place investment cost difference into the dataframe
    sim_df.loc[start_year, "Investment Cost Difference"] = df.loc[start_year, "Investment Cost Difference"]

    # Deactivate maintenance and replacement costs for heating
    if cooling == False:
        sim_df["Maintenance Difference"] = 0
        sim_df["Replacement Difference"] = 0

    else:
        # Calculate the difference in maintenance and replacement costs
        sim_df["Maintenance Difference"] = baseline_maintenance_costs - scenario_maintenance_costs
        sim_df["Replacement Difference"] = baseline_replacement_costs - scenario_replacement_costs
    
    # Create results dataframe for this simulation
    combi_scenario_cooling_df = pd.DataFrame(columns=['cooling_cost_savings'])
    combi_scenario_heating_df = pd.DataFrame(columns=['heating_cost_savings'])
    
    if cooling == True:
        # Test with replace=False
        random_samples_cooling = np.random.choice(c_combi_scenario_samples, size=analysis_period)
        combi_scenario_cooling_df = pd.DataFrame({
            'cooling_cost_savings': np.round(random_samples_cooling, 2)
        }, index=sim_df.index)

    if heating == True:
        # Calculate probabilistic heating costs, test with replace=False
        random_samples_heating = np.random.choice(h_combi_scenario_samples, size=analysis_period)
        heating_savings = np.round(random_samples_heating, 2)
        combi_scenario_heating_df = pd.DataFrame({
            'heating_cost_savings': np.round(random_samples_heating, 2)
        }, index=sim_df.index)

    # Place Operational Savings into the dataframe
    if cooling == True and heating == True:
        sim_df["Operational Savings"] = round(combi_scenario_cooling_df['cooling_cost_savings'] + combi_scenario_heating_df['heating_cost_savings'], 2)
    elif cooling == True and heating == False:
        sim_df["Operational Savings"] = round(combi_scenario_cooling_df['cooling_cost_savings'], 2)
    elif cooling == False and heating == True:
        sim_df["Operational Savings"] = round(combi_scenario_heating_df['heating_cost_savings'], 2)
    else:
        sim_df["Operational Savings"] = 0
    
    # Get a random discount rate from the samples
    discount_rate = analysis_settings["discount_rate"] # np.random.choice(discount_samples)
    discount_factors = [(1 / (1 + discount_rate / 100)) ** n for n in range(analysis_period)]
    years_df = pd.DataFrame({
        "Year Index": list(range(analysis_period)),
        "Year": list(range(start_year, start_year + analysis_period)),
        "Discount Factor": discount_factors
    })
    discount_map = dict(zip(years_df["Year"], years_df["Discount Factor"]))
    # Apply the discount rate to all the columns
    sim_df = apply_discount(sim_df, discount_map)
    
    # Calculate the NPV for each year
    sim_df["NPV"] = sim_df["Operational Savings"] + sim_df["Maintenance Difference"] + sim_df["Replacement Difference"] + sim_df["Investment Cost Difference"]
    
    # Calculate the final NPV
    final_npv = round(sim_df["NPV"].sum(), 2)
    
    # Store the result
    npv_results.append({
        "Index": i,
        "Final NPV": final_npv,
        "Cooling_Savings": combi_scenario_cooling_df['cooling_cost_savings'].tolist() if cooling else [],
        "Heating_Savings": combi_scenario_heating_df['heating_cost_savings'].tolist() if heating else []
    })
    
    print(f"Simulation {i+1}: The final NPV is {final_npv}")

# Convert results to DataFrame and save
npv_df = pd.DataFrame(npv_results)
# Convert the lists to strings for CSV storage
if cooling:
    npv_df['Cooling_Savings'] = npv_df['Cooling_Savings'].apply(lambda x: ','.join(map(str, x)))
if heating:
    npv_df['Heating_Savings'] = npv_df['Heating_Savings'].apply(lambda x: ','.join(map(str, x)))

# Calculate and print statistics
mean_npv = npv_df["Final NPV"].mean()
std_npv = npv_df["Final NPV"].std()
print(f"\nMonte Carlo Results:")
print(f"Mean NPV: {mean_npv:.2f}")
print(f"Standard Deviation: {std_npv:.2f}")
# Central 95% interval of the NPV distribution
lower_bound = np.percentile(npv_df["Final NPV"], 2.5)
upper_bound = np.percentile(npv_df["Final NPV"], 97.5)
# check this line
print(f"95% CI: {lower_bound:.2f} to {upper_bound:.2f} Euro")


# Save statistics to a csv file
statistics_file = os.path.join(current_dir, 'results', folder, f'NPV_statistics.csv')

# Create a dictionary with all the settings and results
statistics_data = {
    'run_index': 0,  # Default value, will be updated if file exists
    'scenario_name': scenario_name,
    'num_simulations': num_simulations,
    'num_pdf_samples': num_pdf_samples,
    'discount_rate': analysis_settings["discount_rate"],
    'cooling_enabled': cooling,
    'heating_enabled': heating,
    'mean_npv': round(mean_npv, 2),
    'std_npv': round(std_npv, 2),
    'lower_bound_95ci': round(lower_bound, 2),
    'upper_bound_95ci': round(upper_bound, 2)
}
if calculate_statistics == True:
    # Convert to DataFrame
    statistics_df = pd.DataFrame([statistics_data])

    # Check if file exists and append or create new
    if os.path.exists(statistics_file):
        # Read existing file
        existing_df = pd.read_csv(statistics_file)
        # Set the run_index to be one more than the maximum existing index
        statistics_df['run_index'] = existing_df['run_index'].max() + 1
        # Append new results
        updated_df = pd.concat([existing_df, statistics_df], ignore_index=True)
        # Save updated results
        updated_df.to_csv(statistics_file, index=False)
    else:
        # Create new file with run_index starting at 0
        statistics_df.to_csv(statistics_file, index=False)

    print(f"\nResults saved to {statistics_file}!")

if save_output == True:
    run_index = statistics_df['run_index'].max()

    if cooling == True and heating == False:
        output_file = os.path.join(current_dir, 'results', folder ,  f'NPV_C_{scenario_name}_{run_index}.csv')
    elif cooling == False and heating == True:
        output_file = os.path.join(current_dir, 'results', folder, f'NPV_H_{scenario_name}_{run_index}.csv')
    elif cooling == True and heating == True:
        output_file = os.path.join(current_dir, 'results', folder, f'NPV_H-C_{scenario_name}_{run_index}.csv')
    else:
        output_file = os.path.join(current_dir, 'results', folder, f'NPV_{scenario_name}_{run_index}.csv')
    npv_df.to_csv(output_file, index=False)

    print(f"\nOutcome saved to {output_file}!")