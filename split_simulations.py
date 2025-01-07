import pandas as pd
import numpy as np

# Function to calculate annual averages for years 2, 5, and 8
def annual_averages(df):
    #year ranges in terms of days (as in malsimgem simulation) 
    year_ranges = {
        "Year_2": (365, 730),
        "Year_5": (1460, 1825),
        "Year_8": (2555, 2920),
    }
    
    # List of columns to calculate annual values for
    columns_to_average = ["prev_true", "EIR_true", "prev_2to10", "inc_2to10", "incall"]
    
    # Dictionary to store annual averages
    annual_averages = []
    
    for run in df['run'].unique():
        run_data = df[df['run'] == run]
        for year, (start, end) in year_ranges.items():
            year_data = run_data[(run_data['t'] >= start) & (run_data['t'] <= end)] #condition for segmentation
            averages = year_data[columns_to_average].mean()
            averages['run'] = run
            averages['year'] = year
            annual_averages.append(averages)
    
    # Dictionaries to DataFrame
    annual_averages_df = pd.DataFrame(annual_averages)
    return annual_averages_df


# Function to filter monthly data for years 10 to 20
def monthly_values(df):
    # Year range in terms of days as in malsimgen
    start, end = 3650, 7320
    monthly_values = df[(df['t'] >= start) & (df['t'] <= end)]
    return monthly_values