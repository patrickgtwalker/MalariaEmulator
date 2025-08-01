import numpy as np
import pandas as pd

#Function to extract required timeseries
def process_dataframe(df):
    """ Here we model the data to compute annual parameters in 2-10 at year 2,5 and 8 as pre-observed data 
    and subsequent monthly parameters as observed data  """
    year_ranges = {
        "Year_2": (390, 720),
        "Year_5": (1470, 1800),
        "Year_8": (2550, 2880),
    }
    processed_data = []
    for run in df['run'].unique():
        run_data = df[df['run'] == run]
        for label, (start, end) in year_ranges.items():
            year_data = run_data[(run_data['t'] >= start) & (run_data['t'] <= end)]
            averages = year_data.select_dtypes(include=[np.number]).mean()
            averages['prev_true'] = averages['prev_2to10']
            averages['incall'] = averages['inc_2to10']
            averages = averages.drop(['prev_2to10', 'inc_2to10'])
            averages['run'] = run
            averages['t'] = end
            processed_data.append(averages)
        monthly_data = run_data[(run_data['t'] >= 3650) & (run_data['t'] <= 7320)]
        monthly_data = monthly_data.drop(columns=['prev_2to10', 'inc_2to10'], errors='ignore')
        processed_data.append(monthly_data)
    return pd.concat([pd.DataFrame([row]) if isinstance(row, pd.Series) else row for row in processed_data], ignore_index=True)
