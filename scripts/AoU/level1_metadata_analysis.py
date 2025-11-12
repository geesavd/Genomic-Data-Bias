# --- Imports ---
# Import the necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery

# --- Setup ---
# Set the style for all our plots
sns.set(style="whitegrid")

# Get the CDR (Curated Data Repository) version for your workspace
# This environment variable is set automatically in your AoU notebook
cdr_dataset_id = os.environ.get('WORKSPACE_CDR')

if cdr_dataset_id:
    print(f"Querying CDR: {cdr_dataset_id}")
else:
    print("Error: WORKSPACE_CDR environment variable not set.")
    # You can manually set this if needed, e.g.:
    # cdr_dataset_id = 'my-project.my_cdr_dataset'

# Initialize the BigQuery client
client = bigquery.Client()

# --- 1. The SQL Query (Demographics) ---
# This query joins the PERSON table with the CONCEPT table to get
# human-readable names for the concept IDs.
# We are only querying Registered Tier fields (Age, Sex, Race, Ethnicity)
# to get this script working for your deadline.

# We use 2025 as the current year to calculate age.
sql_query = f"""
SELECT
    (2025 - p.year_of_birth) AS age,
    c_sex.concept_name AS sex_at_birth,
    c_race.concept_name AS race,
    c_eth.concept_name AS ethnicity
FROM
    `{cdr_dataset_id}.person` p
LEFT JOIN
    `{cdr_dataset_id}.concept` c_sex ON p.sex_at_birth_concept_id = c_sex.concept_id
LEFT JOIN
    `{cdr_dataset_id}.concept` c_race ON p.race_concept_id = c_race.concept_id
LEFT JOIN
    `{cdr_dataset_id}.concept` c_eth ON p.ethnicity_concept_id = c_eth.concept_id
"""

# --- 2. Run Query and Load Data (Demographics) ---
# This runs the query and loads the results directly into a pandas DataFrame.
try:
    print("Running query to fetch demographic data...")
    df_demographics = client.query(sql_query).to_dataframe()
    print(f"Successfully loaded {len(df_demographics)} demographic records.")
    
    # Display the first few rows to check
    print("\nData Head (Demographics):")
    print(df_demographics.head())

except Exception as e:
    print(f"An error occurred: {e}")
    # If the query fails, we'll create an empty DataFrame to avoid further errors
    df_demographics = pd.DataFrame()

# --- 3. Plot the Distributions (Demographics) ---
# We will create a series of plots to visualize the data ratios.

# --- Plot 1: Age Distribution ---
# A histogram is best for a continuous variable like age.
plt.figure(figsize=(10, 6))
# .dropna() handles any participants with a missing year_of_birth
sns.histplot(data=df_demographics.dropna(subset=['age']), x='age', bins=30, kde=True)
plt.title('Distribution of Participant Age', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# --- Plot 2: Sex at Birth ---
# A horizontal count plot is best for long categorical labels.
plt.figure(figsize=(10, 6)) # Adjusted figsize for a good horizontal layout
sns.countplot(data=df_demographics, y='sex_at_birth', order=df_demographics['sex_at_birth'].value_counts().index)
plt.title('Distribution of Sex at Birth', fontsize=16)
plt.xlabel('Count', fontsize=12) # Swapped to be the x-axis label
plt.ylabel('Sex at Birth', fontsize=12) # Swapped to be the y-axis label
# plt.xticks(rotation=45) # This is no longer needed
plt.show()

# --- Plot 3: Race ---
# We use a horizontal plot (y='race') because the labels can be long
# and would overlap if plotted vertically.
plt.figure(figsize=(12, 8))
sns.countplot(data=df_demographics, y='race', order=df_demographics['race'].value_counts().index)
plt.title('Distribution of Self-Reported Race', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Race', fontsize=12)
plt.show()

# --- Plot 4: Ethnicity ---
plt.figure(figsize=(10, 6))
sns.countplot(data=df_demographics, y='ethnicity', order=df_demographics['ethnicity'].value_counts().index)
plt.title('Distribution of Self-Reported Ethnicity', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Ethnicity', fontsize=12)
plt.show()


# --- 4. NEW: Socioeconomic (SES) Analysis ---
# Based on the Data Dictionary, we can link participant 3-digit ZIP codes
# (from the observation table) to the zip3_ses_map table.

print("\n--- Starting Socioeconomic (SES) Analysis ---")

# This concept ID (1585250) corresponds to the "StreetAddress_PIIZIP"
# question from "The Basics" survey, which stores the 3-digit ZIP.
sql_query_ses = f"""
SELECT
    ses.median_income,
    ses.fraction_poverty,
    ses.fraction_no_health_ins
FROM
    `{cdr_dataset_id}.observation` AS obs
JOIN
    `{cdr_dataset_id}.zip3_ses_map` AS ses
ON
    obs.value_as_string = ses.zip3_as_string
WHERE
    obs.observation_source_concept_id = 1585250
"""

# --- 5. Run Query and Load Data (SES) ---
try:
    print("Running query to fetch SES data...")
    df_ses = client.query(sql_query_ses).to_dataframe()
    print(f"Successfully loaded {len(df_ses)} SES records.")
    
    # Display the first few rows to check
    print("\nData Head (SES):")
    print(df_ses.head())

except Exception as e:
    print(f"An error occurred during SES query: {e}")
    df_ses = pd.DataFrame()

# --- 6. Plot the SES Distributions ---

# --- Plot 5: Median Income Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(data=df_ses.dropna(subset=['median_income']), x='median_income', bins=30, kde=True)
plt.title('Distribution of Median Income (by 3-digit ZIP)', fontsize=16)
plt.xlabel('Median Income', fontsize=12)
plt.ylabel('Count of Participants', fontsize=12)
plt.show()

# --- Plot 6: Poverty Fraction Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(data=df_ses.dropna(subset=['fraction_poverty']), x='fraction_poverty', bins=30, kde=True)
plt.title('Distribution of Poverty Percent (by 3-digit ZIP)', fontsize=16)
plt.xlabel('Percent of Population Below Poverty Level (%)', fontsize=12) # <-- CHANGED
plt.ylabel('Count of Participants', fontsize=12)
plt.xlim(0, 60) # <-- ADDED THIS LINE
plt.show()

# --- Plot 7: No Health Insurance Percent Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(data=df_ses.dropna(subset=['fraction_no_health_ins']), x='fraction_no_health_ins', bins=30, kde=True)
plt.title('Distribution of No Health Insurance Percent (by 3-digit ZIP)', fontsize=16)
plt.xlabel('Percent of Population with No Health Insurance (%)', fontsize=12) # <-- CHANGED
plt.ylabel('Count of Participants', fontsize=12)
plt.xlim(0, 40) # <-- ADDED THIS LINE
plt.show()