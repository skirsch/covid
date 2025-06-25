import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Load actual data with explicit dtypes
data = pd.read_csv("jp132101_Koganei-Tokyo_all.csv", low_memory=False, dtype={'age': float, 'date_death': str})

# Event indicator: 1 if date_death is non-empty, 0 otherwise
data['event'] = data['date_death'].notna().astype(int)

# Convert dates to datetime
data['date_death'] = pd.to_datetime(data['date_death'], errors='coerce')
for i in range(1, 8):
    data[f'date_lot{i}'] = pd.to_datetime(data[f'date_lot{i}'], errors='coerce')

# Calculate dose count from non-empty date_lot columns
data['dose_count'] = data[[f'date_lot{i}' for i in range(1, 8)]].notna().sum(axis=1)

# Combine rare dose categories (5, 6, 7 into 5+) and dose_1 with dose_2
data['dose_group'] = data['dose_count'].apply(lambda x: min(x, 5)).astype(str)
data['dose_group'] = data['dose_group'].replace('1', '2')

# Handle age: impute invalid with median
data['age'] = pd.to_numeric(data['age'], errors='coerce')
median_age = data['age'].median()
data['age'] = data['age'].fillna(median_age)
data.loc[data['age'] < 0, 'age'] = median_age
data.loc[data['age'] > 120, 'age'] = median_age

# Bin age into categories
data['age_group'] = pd.cut(data['age'], bins=[0, 60, 120], labels=['<60', '>=60'], include_lowest=True)

# Validate sex
data['sex'] = data['sex'].str.lower().replace({'male': 'm', 'female': 'f'})
data.loc[~data['sex'].isin(['m', 'f']), 'sex'] = 'm'

# Time to event
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 5, 1)

# Calculate time to event
def calculate_time(row):
    end = row['date_death'] if row['event'] == 1 else end_date
    if row['dose_count'] == 0:
        return (end - start_date).days
    else:
        start = row['date_lot1'] if pd.notna(row['date_lot1']) else start_date
        return (end - start).days

data['time'] = data.apply(calculate_time, axis=1)

# Ensure positive time
data = data[data['time'] > 0]

# Covariates
data['sex_binary'] = data['sex'].map({'m': 1, 'f': 0}).astype(int)

# Select columns for Cox model
cox_data = data[['time', 'event', 'age', 'sex_binary', 'dose_group', 'age_group']]

# Calculate average age of deceased per dose group
deceased = data[data['event'] == 1]
avg_age_deceased = deceased.groupby('dose_group', observed=True)['age'].mean()

# Print data summary
print(f"Total records: {len(data)}")
print(f"Number of deaths: {data['event'].sum()}")
print(f"Unvaccinated (0 doses): {len(data[data['dose_count'] == 0])}")
print("\nMissing values before cleaning:")
raw_data = pd.read_csv("jp132101_Koganei-Tokyo_all.csv", low_memory=False, dtype={'age': float, 'date_death': str})
print(raw_data[['age', 'sex', 'date_death']].isna().sum())
print("\nMissing values after cleaning:")
print(data[['age', 'sex', 'dose_group', 'age_group', 'date_death']].isna().sum())
print("\nDropped rows summary:")
dropped_indices = raw_data.index.difference(data.index)
dropped = raw_data.loc[dropped_indices]
print(dropped[['age', 'sex', 'date_death']].describe(include='all'))
print("\nDose distribution:")
print(data['dose_group'].value_counts())
print("\nEvents per dose group:")
print(data.groupby('dose_group', observed=True)['event'].sum())
print("\nEvents per age group:")
print(data.groupby('age_group', observed=True)['event'].sum())
print("\nAverage age of deceased per dose group:")
print(avg_age_deceased)
print("\nCovariate correlations:")
print(cox_data[['age', 'sex_binary']].corr())

# Fit Cox PH model with stratification
cph = CoxPHFitter(penalizer=0.1)
try:
    cph.fit(cox_data, duration_col='time', event_col='event', strata=['dose_group', 'age_group'])
    print("\nCox PH Model Summary:")
    cph.print_summary(decimals=3)
except Exception as e:
    print(f"Cox PH model fitting failed: {e}")

# Check proportional hazards assumption
if hasattr(cph, '_model') and cph._model is not None:
    try:
        ph_test = cph.check_assumptions(cox_data, p_value_threshold=0.05, show_plots=False)
        print("\nProportional Hazards Assumption Check:")
        print(ph_test)
    except Exception as e:
        print(f"Assumption check failed: {e}")

# Logistic Regression Alternative
print("\nLogistic Regression Analysis:")
dummies = pd.get_dummies(data['dose_group'], prefix='dose', drop_first=True, dtype=int)
X = pd.concat([data[['age', 'sex_binary']], dummies], axis=1)
y = data['event']

# Ensure no NaN values
X = X.fillna(0)
y = y[X.index]

# Logistic regression with weaker regularization
model = LogisticRegression(penalty='l2', C=1.0, max_iter=2000, solver='saga')
try:
    model.fit(X, y)
    odds_ratios = np.exp(model.coef_[0])
    # Calculate 95% confidence intervals
    se = np.sqrt(np.diag(model.coef_variance_matrix_))
    ci_lower = np.exp(model.coef_[0] - 1.96 * se)
    ci_upper = np.exp(model.coef_[0] + 1.96 * se)
    print("\nOdds Ratios:", odds_ratios)
    print("95% CI Lower:", ci_lower)
    print("95% CI Upper:", ci_upper)
    print("\nCovariate Names:", X.columns.tolist())
    print("\nConvergence Status:", model.n_iter_)
except Exception as e:
    print(f"Logistic regression failed: {e}")