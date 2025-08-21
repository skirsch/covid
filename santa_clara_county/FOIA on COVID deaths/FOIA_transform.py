import pandas as pd

# Load the original Excel file
file_path = "FOIA_data_request_all_cause_covid_OFFICIAL_FOIA_RESPONSE.xlsx"
all_cause_deaths = pd.read_excel(file_path, sheet_name="all-cause deaths", header=1)
covid_deaths = pd.read_excel(file_path, sheet_name="covid deaths", header=1)

# Function to reshape the data, excluding yearly summary columns
def reshape_data(df, cause_type):
    # Identify and exclude yearly summary columns (e.g., "CY 2018", "CY 2019", etc.)
    quarterly_columns = [col for col in df.columns if col.startswith("Q")]
    df = df[["age_group"] + quarterly_columns]  # Keep only age_group and quarterly columns
    df = df.melt(id_vars=["age_group"], var_name="Quarter", value_name=cause_type)
    df[["Quarter", "Year"]] = df["Quarter"].str.split(" ", expand=True)
    df["Quarter#"] = df["Quarter"].str.extract(r"Q(\d+)").astype(int)
    df["Year"] = df["Year"].astype(int)
    df = df.rename(columns={"age_group": "Age range"})
    return df

# Reshape all-cause deaths and COVID deaths
all_cause_reshaped = reshape_data(all_cause_deaths, "All cause deaths")
covid_reshaped = reshape_data(covid_deaths, "COVID deaths")

# Merge the two datasets on Year, Quarter#, and Age range
merged_df = pd.merge(all_cause_reshaped, covid_reshaped, on=["Year", "Quarter#", "Age range"], how="outer")

# Sort the data by Year, Quarter#, and Age range
merged_df = merged_df.sort_values(by=["Year", "Quarter#", "Age range"])

# Save the new DataFrame to an Excel file
output_file = "Transformed_Deaths_Data.xlsx"
merged_df.to_excel(output_file, index=False)

print(f"New file saved as {output_file}")
