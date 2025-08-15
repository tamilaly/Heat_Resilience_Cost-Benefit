# Probabilistic Cost-Benefit Assessment for Renovation Measures targeting heating and cooling efficiency

This script can be used to simulate the possible chances for cost recovery and the development of the total cost in the long term. The goal might be to understand the likelihood and impact of worst-case scenarios. The input data is based on energy plus simulations 
historical weather data analysis used as an

## Getting Started

### Dependencies

* Data analysis and manipulation (pandas, numpy)
* Statistical analysis and probability distributions (scipy)
* Data visualization (matplotlib)
* File I/O operations (json, csv, os)

## Installing packages

Using pip installer: 
```
pip install pandas matplotlib numpy scipy
```
### Executing program

Using the code Net Present Value and Total Cost of Ownership can be calculated probabilistically based on climate variability. Also, probability density for performance indicators such as energy use intensity, carbon intensity, and heat resilience indicators can be estimated. The operations are all separated into several programs. The basis for the calculation is the energy simulation results for several historical weather years representing certain points on the cumulative probability function e.g. 2019: 51%. 

#### configurations: use of analysis settings
* universal_data.json stores all fixed data inputs for energy pricing etc. 
* analysis_settings.json stores the specific settings for the cost-benefit such as the discount rate, analysis period, but also the reference yeas and percentages to reconstruct the distribution

#### scenarios: renovation scenario & baseline
the json files are the backlog to all necessary data for the scenarios
baseline_scenario.json stores the link to the estimated energy demands e.g. "cooling_file": "AC_Simple-01_extra_CDD.csv" 
and all the required heating or cooling system data as in efficiency rate, supply source, global costs
* renovation_scenarios.json
* same goes here! scenarios are concatenated. 
The available scenarios are:
    * S01_simple_shading
    * S02_simple_paint
    * S04_simple_combi
    * M01_medium_shading
    * D01_deep_shading-paint

Create your own scenario by specifying the following
system types: air_conditioning_split, gas_condensing_boiler, district_heating
energy sources: districtheating, gas, electricity

```
"Scenario_Name": {
    "household_size": 0,
    "file": "optional_file_not_used.csv",
    "cooling_file": "scenario-name_CDD.csv",
    "heating_file": "scenario-name_HDD.csv",
    "overheating_file": "Scenario_Comfort.csv",
    "cooling_energy_source": "source_name",
    "heating_energy_source": "source_name",
    "cooling_system_type": "system_name",
    "heating_system_type": "system_name",
    "cooling_cop_or_eer": 1.0,
    "heating_cop_or_eer": 1.0,
    "investment_cost_cooling": 0,
    "investment_cost_heating": 0,
    "maintenance_cost_item1": 0,
    "maintenance_cycle_item1": 0,
    "replacement_cost_item1": 0,
    "replacement_cycle_item1": 0
    }
```

#### TCOO estimation with Monte Carlo simulation

The settings for scenario to analyze, run times, number of samples and which operating cost to account for heating or cooling are defined at the top of the program. 

#### TCOO graphs

after simulation the results stored can be visualized in graphs. 

#### NPV estimation with Monte Carlo simulation

##### generating a fixed set of savings samples

1. first step to NPV simulation is generating the sample set of savings representative of the energy savings distributions

##### NPV estimation

NPV is different from TCOO
NPV calculation with fixed samples


### '_Performance' is a collection of scripts to evaluate the distribution statistics for energy, carbon, cost, and also overheating indicators

## Help

///

## Authors

Contributors names and contact info

ex. Tamara Lalyko [@tamilaly]






