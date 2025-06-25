import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

# Load and preprocess
df = pd.read_csv("jp132101_Koganei-Tokyo_all.csv", low_memory=False)  # replace with your actual filename
for i in range(1, 8):
    df[f'date_lot{i}'] = pd.to_datetime(df[f'date_lot{i}'], errors='coerce')
df['date_death'] = pd.to_datetime(df['date_death'], errors='coerce')

# Build risk data
risk_records = []
for i in range(1, 8):
    col = f'date_lot{i}'
    mask = (df[col].notna()) & (df['date_death'].notna()) & (df[col] <= df['date_death'])
    sub = df[mask].copy()
    sub['dose_number'] = i
    sub['time_at_risk_days'] = (sub['date_death'] - sub[col]).dt.days
    risk_records.append(sub[['age', 'sex', 'dose_number', 'time_at_risk_days']])

df_risk = pd.concat(risk_records)
df_risk['sex_encoded'] = df_risk['sex'].map({'m': 0, 'f': 1})
df_risk['event'] = 1

# Fit Cox model
duration = df_risk['time_at_risk_days'].values
event = df_risk['event'].values
covariates = df_risk[['dose_number', 'age', 'sex_encoded']]
model = PHReg(duration, covariates, status=event)
results = model.fit()
print(results.summary())


"""import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from datetime import datetime

# Load actual data
data = pd.read_csv("jp132101_Koganei-Tokyo_all.csv", low_memory=False)

# Event indicator: 1 if date_death is non-empty, 0 otherwise
data['event'] = data['date_death'].notna().astype(int)

# Convert dates to datetime
data['date_death'] = pd.to_datetime(data['date_death'], errors='coerce')
for i in range(1, 8):
    data[f'date_lot{i}'] = pd.to_datetime(data[f'date_lot{i}'], errors='coerce')

# Time to event
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 5, 1)

# Prepare time-dependent data
time_dep_data = []

for _, row in data.iterrows():
    # Get vaccination dates
    vacc_dates = [row[f'date_lot{i}'] for i in range(1, 8) if pd.notna(row[f'date_lot{i}'])]
    vacc_dates.sort()  # Ensure chronological order
    dose_count = len(vacc_dates)
    
    # Define intervals
    if dose_count == 0:
        # Unvaccinated: Single interval
        end = row['date_death'] if row['event'] == 1 else end_date
        time_dep_data.append({
            'id': row['id'],
            'start_time': 0,
            'stop_time': (end - start_date).days,
            'event': row['event'],
            'age': row['age'],
            'sex_binary': 1 if row['sex'] == 'm' else 0,
            'dose_count': 0
        })
    else:
        # Vaccinated: Multiple intervals
        prev_time = start_date
        for i, vacc_date in enumerate(vacc_dates):
            # Interval from previous time to current vaccination
            time_dep_data.append({
                'id': row['id'],
                'start_time': (prev_time - start_date).days,
                'stop_time': (vacc_date - start_date).days,
                'event': 0,  # No event during this interval
                'age': row['age'],
                'sex_binary': 1 if row['sex'] == 'm' else 0,
                'dose_count': i
            })
            prev_time = vacc_date
        
        # Final interval: from last vaccination to death/censoring
        end = row['date_death'] if row['event'] == 1 else end_date
        time_dep_data.append({
            'id': row['id'],
            'start_time': (prev_time - start_date).days,
            'stop_time': (end - start_date).days,
            'event': row['event'],
            'age': row['age'],
            'sex_binary': 1 if row['sex'] == 'm' else 0,
            'dose_count': dose_count
        })

# Create DataFrame
cox_data = pd.DataFrame(time_dep_data)

# Ensure positive time intervals
cox_data = cox_data[cox_data['stop_time'] > cox_data['start_time']]

# Convert dose_count to categorical (dummy variables)
cox_data['dose_count'] = cox_data['dose_count'].astype(str)
dummies = pd.get_dummies(cox_data['dose_count'], prefix='dose', drop_first=True)  # 0 doses as reference
cox_data = pd.concat([cox_data, dummies], axis=1)

# Select columns for Cox model
covariates = ['age', 'sex_binary'] + [col for col in cox_data.columns if col.startswith('dose_')]
cox_data = cox_data[['start_time', 'stop_time', 'event'] + covariates]

# Fit Cox PH model with time-dependent covariates
cph = CoxPHFitter()
try:
    cph.fit(cox_data, duration_col='stop_time', event_col='event', entry_col='start_time')
    # Print summary
    cph.print_summary(decimals=3)
except Exception as e:
    print(f"Model fitting failed: {e}")

# Check proportional hazards assumption
if cph._model is not None:
    try:
        ph_test = cph.check_assumptions(cox_data, p_value_threshold=0.05, show_plots=False)
        print("\nProportional Hazards Assumption Check:")
        print(ph_test)
    except Exception as e:
        print(f"Assumption check failed: {e}")"""