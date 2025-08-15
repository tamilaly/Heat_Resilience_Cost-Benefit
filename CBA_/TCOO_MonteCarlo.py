import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy import stats

# The purpose this script is to calculate the most likely total cost of ownership (TCOO) for a given scenario
# The TCOO is the sum of the investment cost, the operational cost, the maintenance cost and the replacement cost
# The script will calculate the TCOO for a given scenario (also baseline)
# The calculated TCOO are saved in a csv file including the corresponding energy/operational cost samples
# additionally for every run of the Monte Carlo simulation the results are added to a general statistics csv file

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# SETTINGS
# Scenario to simulate
scenario_name = "S04_simple_combi"
cooling = True # True or False
heating = False # True or False

# Number of simulations for Monte Carlo
num_simulations = 1000 # number of simulations  
num_pdf_samples = 10000 # number of samples for the probability density function

save_output = True # True or False
calculate_statistics = True # True or False
# output folder to save to named after the current date
folder = pd.Timestamp.now().strftime('%y-%m-%d')

# START OF THE SCRIPT
# Load the output file
if cooling == True and heating==False:
    output_file = os.path.join(current_dir, 'results', folder, f'TCOO_{scenario_name}_C.csv')
elif cooling == False and heating==True:
    output_file = os.path.join(current_dir, 'results', folder, f'TCOO_{scenario_name}_H.csv')
elif cooling == True and heating==True:
    output_file = os.path.join(current_dir, 'results', folder, f'TCOO_{scenario_name}_H-C.csv')
else:
    # dont save the output print (error)
    print('unclear input variables given')

# Create the output directory if it doesn't exist
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

accumulated_tcoo = 0

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

if scenario_name == 'baseline':
    scenario = baseline_data
else: 
    scenario  = renovation_scenarios[scenario_name]

def generate_kde_samples(scenario_data, universal_data, variable_name, num_samples=1000, percentiles=[0, 5, 25, 50, 75, 95, 100]):
    """
    Generate KDE samples from cooling demand data.
    
    Args:
        scenario_data (dict): Dictionary containing scenario information including cooling_file and cooling_system_type
        universal_data (dict): Dictionary containing universal parameters including cooling COP values
        num_samples (int): Number of samples to generate
        percentiles (list): List of percentiles to use for interpolation
        
    Returns:
        numpy.ndarray: Array of KDE samples
    """
    file_key = f"{variable_name}_file"
    # Read values from demand file
    scenario_file = os.path.join(current_dir, 'scenarios', scenario_data[file_key])
    scenario_df = pd.read_csv(scenario_file)
    
    # Capitalize the variable name
    key = f"{variable_name.capitalize()}_Annual_Total"
    # Sort values
    sorted_scenario_df = scenario_df.sort_values(by=key)
    sorted_scenario_values = sorted_scenario_df[key].values
    
    # Get the cooling COP from the scenario
    if variable_name == "cooling":
        scenario_cop = universal_data[f"{scenario_data['cooling_system_type']}_cop"]
    elif variable_name == "heating":
        scenario_cop = universal_data[f"{scenario_data['heating_system_type']}_eer"]
    
    # Calculate cooling energy use
    scenario_energy_use = sorted_scenario_values / scenario_cop
    
    if variable_name == "heating":
        # Add fixed costs for the heating source
        s_fixed_cost_key = f"fixed_cost_{scenario_data['heating_energy_source']}"
        s_fixed_cost = universal_data[s_fixed_cost_key]
        s_total_fixed_cost = s_fixed_cost * scenario_data['household_size']

        # Get heating energy price
        s_heating_energy_source = scenario_data['heating_energy_source']
        s_price_per_kwh = universal_data[f"price_{s_heating_energy_source}"]
        # calculate heating cost
        s_cost = scenario_energy_use * s_price_per_kwh + s_total_fixed_cost
    else:
        # Get cooling energy price
        s_cooling_energy_source = scenario_data['cooling_energy_source']
        s_price_per_kwh = universal_data[f"price_{s_cooling_energy_source}"]
        # calculate cooling cost
        s_cost = scenario_energy_use * s_price_per_kwh
    
    # Convert percentiles to probabilities
    probabilities = np.array(percentiles) / 100.0
    
    # Create interpolation function
    interp_func = interp1d(probabilities, s_cost, bounds_error=False, fill_value="extrapolate")
    
    # Generate uniform random samples and transform them
    uniform_samples = np.random.uniform(0, 1, 5000)
    samples = interp_func(uniform_samples)
    
    # Fit KDE and generate samples
    kde = gaussian_kde(samples, bw_method=0.6)
    
    # Apply bounds
    kde_samples = kde.resample(num_samples)[0]
    # delete the negative values
    min_bound_scenario = min(s_cost)
    # max_bound_scenario = max(s_cost)*1.1

    kde_samples = kde_samples[kde_samples > min_bound_scenario]
    # kde_samples = kde_samples[kde_samples < max_bound_scenario]

    return kde,kde_samples

# Apply discounting
def apply_discount(df, discount_map):
    discounted = df.copy()
    for year in discounted.index:
        discounted.loc[year] *= discount_map[year]
    return discounted

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

# Source the analysis settings
start_year = analysis_settings["start_year"]
reference_year = analysis_settings["reference_year"]
analysis_period = analysis_settings["analysis_period"]
years = list(range(start_year, start_year + analysis_period))

# Get a random discount rate from the samples
discount_rate = analysis_settings["discount_rate"] # np.random.choice(discount_samples)
discount_factors = [(1 / (1 + discount_rate / 100)) ** n for n in range(analysis_period)]
years_df = pd.DataFrame({
    "Year Index": list(range(analysis_period)),
    "Year": list(range(start_year, start_year + analysis_period)),
    "Discount Factor": discount_factors
})

# Set up the dataframe
df = pd.DataFrame(0.0, index=years, columns=["Investment Cost", "Operational Cost", "Maintenance Cost", "Replacement Cost"])

if cooling == True:
    # Calculate the investment cost difference
    df.loc[start_year, "Investment Cost"] = scenario["investment_cost_cooling"]
    # source percentiles from analysis_settings
    percentiles = analysis_settings["reference_percentages_cooling"]
    # Make a combined savings cost dataframe for COOLING
    c_pdf, c_combi_scenario_samples = generate_kde_samples(scenario, universal_data, variable_name="cooling", num_samples=num_pdf_samples, percentiles=percentiles)
   
if heating == True:
    # Calculate the investment cost difference
    df.loc[start_year, "Investment Cost"] += scenario["investment_cost_heating"]
    # source percentiles from analysis_settings
    percentiles = analysis_settings["reference_percentages_heating"]
    # Make a combined savings cost dataframe for HEATING
    h_kde, h_combi_scenario_samples = generate_kde_samples(scenario, universal_data, variable_name="heating", num_samples=num_pdf_samples, percentiles=percentiles)

# calculate maintenance and replacement costs
scenario_maintenance_costs = calculate_maintenance_cost(scenario, start_year, analysis_period) # scenario maintenance costs
scenario_replacement_costs = calculate_replacement_cost(scenario, start_year, analysis_period) # scenario replacement costs

# Monte Carlo Simulation
tcoo_results = []

for i in range(num_simulations):
    
    # Create a fresh DataFrame for each simulation
    sim_df = pd.DataFrame(0.0, index=years, columns=["Investment Cost", "Operational Cost", "Maintenance Cost", "Replacement Cost"])

    # Place investment coste into the dataframe
    sim_df.loc[start_year, "Investment Cost"] = df.loc[start_year, "Investment Cost"]

    # Deactivate maintenance and replacement costs for heating: This is problematic for Medium and deep scenarios?
    if cooling == False:
        sim_df["Maintenance Cost"] = 0
        sim_df["Replacement Cost"] = 0

    else:
        # Calculate the cost for maintenance and replacement
        sim_df["Maintenance Cost"] = scenario_maintenance_costs
        sim_df["Replacement Cost"] = scenario_replacement_costs
    
    # Create results dataframe for this simulation
    combi_scenario_cooling_df = pd.DataFrame(columns=['cooling_cost'])
    combi_scenario_heating_df = pd.DataFrame(columns=['heating_cost'])
    
    if cooling == True:
        # Test with replace=False
        random_samples_cooling = np.random.choice(c_combi_scenario_samples, size=analysis_period)
        combi_scenario_cooling_df = pd.DataFrame({
            'cooling_cost': np.round(random_samples_cooling, 2)
        }, index=sim_df.index)

    if heating == True:
        # Calculate probabilistic heating costs, test with replace=False
        random_samples_heating = np.random.choice(h_combi_scenario_samples, size=analysis_period)
        combi_scenario_heating_df = pd.DataFrame({
            'heating_cost': np.round(random_samples_heating, 2)
        }, index=sim_df.index)

    # Place Operational Savings into the dataframe
    if cooling == True:
        sim_df["Operational Cost"] += combi_scenario_cooling_df['cooling_cost']
    if heating == True:
        sim_df["Operational Cost"] += combi_scenario_heating_df['heating_cost']
    
    discount_map = dict(zip(years_df["Year"], years_df["Discount Factor"]))
    # Apply the discount rate to all the columns
    sim_df = apply_discount(sim_df, discount_map)

    # Calculate the Cost for each year
    sim_df["cost_sum"] = sim_df["Operational Cost"] + sim_df["Maintenance Cost"] + sim_df["Replacement Cost"] + sim_df["Investment Cost"]


    # Calculate the final NPV
    tcoo = round(sim_df["cost_sum"].sum(), 2)
        # Store the result
    tcoo_results.append({
        "Index": i,
        "TCOO": tcoo,
        "Cooling_Cost": combi_scenario_cooling_df['cooling_cost'].tolist() if cooling else [],
        "Heating_Cost": combi_scenario_heating_df['heating_cost'].tolist() if heating else []
    })
    
    print(f"Simulation {i+1}: The final 'total cost of ownership' for {scenario_name} is {tcoo}")

# Convert results to DataFrame and save
tcoo_df = pd.DataFrame(tcoo_results)
# Convert the lists to strings for CSV storage
if cooling:
    tcoo_df['Cooling_Cost'] = tcoo_df['Cooling_Cost'].apply(lambda x: ','.join(map(str, x)))
if heating:
    tcoo_df['Heating_Cost'] = tcoo_df['Heating_Cost'].apply(lambda x: ','.join(map(str, x)))

if save_output == True:
    tcoo_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if calculate_statistics == True:

    # Calculate and print statistics
    mean_tcoo = tcoo_df["TCOO"].mean()
    std_tcoo = tcoo_df["TCOO"].std()
    print(f"\nMonte Carlo Results: {scenario_name}")
    print(f"Mean TCOO: {mean_tcoo:.2f}")
    print(f"Standard Deviation: {std_tcoo:.2f}")

    # Central 95% interval of the NPV distribution
    lower_bound = np.percentile(tcoo_df["TCOO"], 2.5)
    upper_bound = np.percentile(tcoo_df["TCOO"], 97.5)
    # check this line
    print(f"95% CI: {lower_bound:.2f} to {upper_bound:.2f} Euro")

    # Save statistics to a csv file
    statistics_file = os.path.join(current_dir, 'results', folder, f'TCOO_statistics.csv')

    # Create a dictionary with all the settings and results
    statistics_data = {
        'run_index': 0,  # Default value, will be updated if file exists
        'scenario_name': scenario_name,
        'num_simulations': num_simulations,
        'num_pdf_samples': num_pdf_samples,
        'discount_rate': analysis_settings["discount_rate"],
        'cooling_enabled': cooling,
        'heating_enabled': heating,
        'mean_tcoo': round(mean_tcoo, 2),
        'std_tcoo': round(std_tcoo, 2),
        'lower_bound_95ci': round(lower_bound, 2),
        'upper_bound_95ci': round(upper_bound, 2)
    }

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
        print(f"\nResults saved to: {statistics_file}")
    else:
        # Create new file with run_index starting at 0
        statistics_df.to_csv(statistics_file, index=False)
        print(f"\nResults saved to: {statistics_file}")