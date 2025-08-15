import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import json

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

def plot_overheating_data(scenario_names, variable, values):
    # remove '-extra' from scenario names
    scenario_names = [name.replace('-extra', '') for name in scenario_names]
    
    plt.figure(figsize=(12, 6))

    colors = [COLOR_0, COLOR_1, COLOR_2, COLOR_3]

    # make bar plot; set min to 0; column width to 0.5
    bars = plt.bar(scenario_names, values, color=[plt.matplotlib.colors.to_rgba(c, alpha=0.7) for c in colors], width=0.5)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    # Make a stirng from variable with spaces between words
    variable_str = ' '.join(word.capitalize() for word in variable.split('_'))
    plt.ylabel(variable_str)
    plt.title(f'{variable_str} of 2020 for each scenario')
    plt.show()

def plot_double_bars(scenario_names, variable_1, values_1, variable_2, values_2):
    # remove '-extra' from scenario names
    scenario_names = [name.replace('-extra', '') for name in scenario_names]
    
    plt.figure(figsize=(14, 10))

    colors = [COLOR_0, COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5]

    # Set the positions for the bars
    x = np.arange(len(scenario_names))
    width = 0.35  # Width of the bars
    
    # Create the bars side by side
    bars_1 = plt.bar(x - width/2, values_1, width, color=colors, label=variable_1)
    bars_2 = plt.bar(x + width/2, values_2, width, color=[plt.matplotlib.colors.to_rgba(c, alpha=0.7) for c in colors], label=variable_2)
    
    # Set the x-axis ticks and labels
    plt.xticks(x, scenario_names)
    # Add value labels on top of each bar
    for bar in bars_1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    for bar in bars_2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    # Make a string from variable with spaces between words
    variable_str_1 = ' '.join(word for word in variable_1.split('_'))
    variable_str_2 = ' '.join(word for word in variable_2.split('_'))
    plt.title(f'{variable_str_1} and {variable_str_2}\n in 2020 for each scenario')
    # Add legend at the bottom
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')

    # tilt x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.show()

def plot_bars_with_line(scenario_names, variable_1, values_1, variable_2, values_2):
    # remove '-extra' from scenario names
    scenario_names = [name.replace('-extra', '') for name in scenario_names]
    
    plt.figure(figsize=(12, 6))

    colors = [Color_0, COLOR_1, COLOR_2, COLOR_3]

    # Set the positions for the bars
    x = np.arange(len(scenario_names))
    width = 0.35  # Width of the bars
    
    # Create the bars side by side
    bars_1 = plt.bar(x, values_1, width, color=colors, label=variable_1)
    # Create line plot for second variable
    line_2 = plt.plot(x, values_2, marker='o', linestyle='--', color='black', label=variable_2)

    # Set the x-axis ticks and labels
    plt.xticks(x, scenario_names)
    # Add value labels on top of each bar
    for bar in bars_1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    # Make a string from variable with spaces between words
    variable_str_1 = ' '.join(word for word in variable_1.split('_'))
    variable_str_2 = ' '.join(word for word in variable_2.split('_'))
    plt.title(f'{variable_str_1} and\n{variable_str_2}\nof 2020 for each scenario')
    # Add legend at the bottom
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    
    plt.show()

def plot_double_bars_with_line(scenario_names, variable_1, values_1, variable_2, values_2, variable_3, values_3):
    # remove '-extra' from scenario names
    scenario_names = [name.replace('-extra', '') for name in scenario_names]
    
    plt.figure(figsize=(14, 10))

    colors = [COLOR_0, COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5]

    # Set the positions for the bars
    x = np.arange(len(scenario_names))
    width = 0.35  # Width of the bars
    
    # Create the bars side by side
    bars_1 = plt.bar(x-width/2, values_1, width, color=colors, label=variable_1)
    bars_2 = plt.bar(x+width/2, values_2, width, color=[plt.matplotlib.colors.to_rgba(c, alpha=0.7) for c in colors], label=variable_2)	

    # Create line plot for third variable
    plt.scatter(x, values_3,marker='o', color='black', label=variable_3)
    # add strips to the line
    for i in range(len(x)):
        plt.plot([x[i]-0.35, x[i]+0.35], [values_3[i], values_3[i]], color='black', linewidth=2)
    
    # plot a continuous line at height of 1 extending beyond graph edges
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1)

    # Set the x-axis ticks and labels
    plt.xticks(x, scenario_names)

    # Make a string from variable with spaces between words
    variable_str_1 = ' '.join(word for word in variable_1.split('_'))
    variable_str_2 = ' '.join(word for word in variable_2.split('_'))
    variable_str_3 = ' '.join(word for word in variable_3.split('_'))
    plt.title(f'{variable_str_1} and {variable_str_2} and\n{variable_str_3} of 2020 for each scenario')
    # Add legend at the bottom
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')

    # tilt x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.show()


# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

scenario_name_0 = "baseline"
scenario_name_1 = "S01_simple_shading-extra"
scenario_name_2 = "S02_simple_paint"
scenario_name_3 = "S04_simple_combi"
scenario_name_4 = "M01_medium_shading"
scenario_name_5 = "D01_deep_shading-paint"
scenario_name_6 = "D02_deep_brick-veneer"

# concat scenario names
scenario_names = [scenario_name_0, scenario_name_1, scenario_name_2, scenario_name_3, scenario_name_4, scenario_name_5]

# load scenarios 
with open(os.path.join(current_dir, 'scenarios', 'renovation_scenarios.json')) as f:
    scenarios = json.load(f)

# csv scenario comfort file path
scenario_comfort = os.path.join(current_dir, 'scenarios', 'Scenario_Comfort.csv')

df_comfort = pd.read_csv(scenario_comfort)

# Get the overheating data for each scenario: 
variable_1 = "Overheated_Hours_28_SET"
variable_2 = "Degree_Hours_28_SET"
variable_3 = "Indoor_Overheating_Degree"
variable_4 = "Ambient_Warmness_Degree"
variable_5 = "Overheating_Escalation_Factor"

# Collect values for the first two variables
values_1 = []  # Overheated_Hours_28_SET
values_2 = []  # Degree_Hours_28_SET
values_3 = []  # Indoor_Overheating_Degree
values_4 = []  # Ambient_Warmness_Degree
values_5 = []  # Overheating_Escalation_Factor

for scenario in scenario_names:
    # Get the row for this scenario
    scenario_data = df_comfort[df_comfort['scenario'] == scenario]
    if not scenario_data.empty:
        values_1.append(scenario_data[variable_1].values[0])
        values_2.append(scenario_data[variable_2].values[0])
        values_3.append(scenario_data[variable_3].values[0])
        values_4.append(scenario_data[variable_4].values[0])
        values_5.append(scenario_data[variable_5].values[0])

# Create the double bar plot
# plot_double_bars_with_line(scenario_names, variable_3, values_3, variable_4, values_4, variable_5, values_5)
plot_double_bars(scenario_names, variable_1, values_1, variable_2, values_2)


"""values_1 = []

for scenario_name in scenario_names:
    data = scenario_results[scenario_name][variable_1]
    print(data)
    values_1.append(data)

values_2 = []
# Get the data for each scenario
for scenario_name in scenario_names:
    data = scenario_results[scenario_name][variable_2]
    print(data)
    values_2.append(data)

values_3 = []

for scenario_name in scenario_names:
    data = scenario_results[scenario_name][variable_3]
    print(data)
    values_3.append(data)

values_4 = []

for scenario_name in scenario_names:
    data = scenario_results[scenario_name][variable_4]
    print(data)
    values_4.append(data)


# plot_double_bars_with_line(scenario_names, variable_2, values_2, variable_3, values_3, variable_4, values_4)

plot_overheating_data(scenario_names, variable_1, values_1)"""
