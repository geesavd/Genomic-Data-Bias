# --- Imports ---
# Import the necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
import scipy.stats as stats
import scikit_posthocs as sp

# --- Setup ---
# Set the style for all our plots
sns.set(style="whitegrid")

# Get the CDR (Curated Data Repository) version for your workspace
cdr_dataset_id = os.environ.get('WORKSPACE_CDR')

if cdr_dataset_id:
    print(f"Querying CDR: {cdr_dataset_id}")
else:
    print("Error: WORKSPACE_CDR environment variable not set.")
    # This block should not be reached in a functioning AoU notebook.

# Initialize the BigQuery client
client = bigquery.Client()

# --- 1. THE COMBINED SQL QUERY (LEVEL 2 ANALYSIS) ---
# This single query gets BOTH demographics and SES data in one table.
# We use LEFT JOINs so we can also analyze "missingness."
# We link person -> observation (for 3-digit zip) -> zip3_ses_map (for SES data)

sql_query = f"""
SELECT
    p.person_id,
    (2025 - p.year_of_birth) AS age,
    c_sex.concept_name AS sex_at_birth,
    c_race.concept_name AS race,
    c_eth.concept_name AS ethnicity,
    ses.median_income,
    ses.fraction_poverty,
    ses.fraction_no_health_ins
FROM
    `{cdr_dataset_id}.person` p
LEFT JOIN
    `{cdr_dataset_id}.concept` c_sex ON p.sex_at_birth_concept_id = c_sex.concept_id
LEFT JOIN
    `{cdr_dataset_id}.concept` c_race ON p.race_concept_id = c_race.concept_id
LEFT JOIN
    `{cdr_dataset_id}.concept` c_eth ON p.ethnicity_concept_id = c_eth.concept_id
LEFT JOIN
    `{cdr_dataset_id}.observation` AS obs
ON
    p.person_id = obs.person_id AND obs.observation_source_concept_id = 1585250 -- 1585250 is the concept ID for 3-digit ZIP
LEFT JOIN
    `{cdr_dataset_id}.zip3_ses_map` AS ses
ON
    obs.value_as_string = ses.zip3_as_string
"""

# --- 2. Run Query and Load Data ---
try:
    print("Running combined query to fetch all analysis data...")
    df_analysis = client.query(sql_query).to_dataframe()
    print(f"Successfully loaded {len(df_analysis)} records.")
    
    # Display the first few rows to check
    print("\nData Head (Combined Analysis):")
    print(df_analysis.head())

except Exception as e:
    print(f"An error occurred: {e}")
    df_analysis = pd.DataFrame()


# --- 3. LEVEL 2 ANALYSIS (The *Meaningful* Analysis for Your Paper) ---

# --- Analysis Plot 1: Income vs. Race ---
# This boxplot shows if income is differently distributed across racial groups.

plt.figure(figsize=(14, 8))

# 1. Create a filtered DataFrame that REMOVES the 'No matching concept' group
plot_data = df_analysis[df_analysis['race'] != 'No matching concept']

# 2. Use this 'plot_data' for the boxplot. We will plot all outliers.
sns.boxplot(data=plot_data, x='race', y='median_income')

# --- BEGIN PLOT FIX ---
# 3. Set the Y-axis limit to 0-180,000 as requested
#    This will "zoom in" on the boxes and cut off the most extreme outliers.
plt.ylim(0, 180000)
# --- END PLOT FIX ---

plt.title('Median Income Distribution by Self-Reported Race', fontsize=16)
plt.xlabel('Self-Reported Race', fontsize=12)
plt.ylabel('Median Income (by 3-digit ZIP)', fontsize=12)
plt.xticks(rotation=45, ha='right')

# --- BEGIN STATISTICAL TEST ---
# We use a Kruskal-Wallis H-test.
# This test compares MEDIANS, not means, and is robust to outliers.

# 1. Clean data for the test (remove NaNs AND 'No matching concept')
test_data = df_analysis.dropna(subset=['race', 'median_income'])
test_data = test_data[test_data['race'] != 'No matching concept'] 

# 2. Get a list of all unique race groups
groups = test_data['race'].unique()

# 3. Create a list of the median_income data for each group
income_by_group = [test_data['median_income'][test_data['race'] == group] for group in groups]

# 4. Run the Kruskal-Wallis H-test
hvalue, pvalue = stats.kruskal(*income_by_group)

# 5. Create the text to display on the plot
if pvalue < 0.05:
    stat_text = "Statistically Significant Deviation\n"
    box_color = 'lightgreen'
    
    # Format the p-value string to handle very small numbers
    if pvalue < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {pvalue:.3f}"
    
    stat_text += f"(Kruskal-Wallis {p_text})"
    
else:
    box_color = 'lightgray'
    stat_text = f"No Significant Deviation Found\n(Kruskal-Wallis p = {pvalue:.3f})"

# --- BEGIN TEXT BOX FIX ---
# 6. Add the text to the TOP-RIGHT corner to avoid data
plt.text(0.98, 0.98, stat_text,
         transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='right', # <-- Set horizontal to 'right'
         bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.5))
# --- END TEXT BOX FIX ---
         
# --- END STATISTICAL TEST ---

plt.show()





# --- Data Cleaning for Plot 2 Label ---
# We replace the long, messy label with a clean one *before* plotting.
long_label = 'What Race Ethnicity: Race Ethnicity None Of These'
short_label = 'None Of These'
df_analysis['ethnicity'] = df_analysis['ethnicity'].replace(long_label, short_label)


# --- Analysis Plot 2: Poverty vs. Ethnicity ---
# This boxplot shows if poverty is differently distributed by ethnicity.
plt.figure(figsize=(10, 6))

# 1. Use the *full* DataFrame for plotting.
sns.boxplot(data=df_analysis, x='ethnicity', y='fraction_poverty')

# 2. Set the Y-axis limit to "zoom in" on the boxes.
plt.ylim(0, 75) # Zoom in on the 0-75% range

plt.title('Poverty Percent Distribution by Self-Reported Ethnicity', fontsize=16)
plt.xlabel('Self-Reported Ethnicity', fontsize=12)
plt.ylabel('Poverty Percent (by 3-digit ZIP)', fontsize=12)
plt.xticks(rotation=45, ha='right')

# --- BEGIN STATISTICAL TEST ---
# We use a Kruskal-Wallis H-test (compares medians, robust to outliers).

# 1. Clean data for the test (remove NaNs ONLY)
test_data_eth = df_analysis.dropna(subset=['ethnicity', 'fraction_poverty'])

# 2. Get a list of all unique ethnicity groups
groups_eth = test_data_eth['ethnicity'].unique()

# 3. Create a list of the poverty data for each group
poverty_by_group = [test_data_eth['fraction_poverty'][test_data_eth['ethnicity'] == group] for group in groups_eth]

# 4. Run the Kruskal-Wallis H-test
hvalue, pvalue = stats.kruskal(*poverty_by_group)

# 5. Create the text to display on the plot
if pvalue < 0.05:
    stat_text = "Statistically Significant Deviation\n"
    box_color = 'lightgreen'
    
    # Format the p-value string to handle very small numbers
    if pvalue < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {pvalue:.3f}"
    
    stat_text += f"(Kruskal-Wallis {p_text})"
    
else:
    box_color = 'lightgray'
    stat_text = f"No Significant Deviation Found\n(Kruskal-Wallis p = {pvalue:.3f})"

# 6. Add the text to the TOP-RIGHT corner to avoid data
plt.text(0.98, 0.98, stat_text,
         transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.5))
         
# --- END STATISTICAL TEST ---

plt.show()

# --- BEGIN POST-HOC TEST (Dunn's Test) ---
# This test will tell us *which* groups are different from each other.

print("\n--- Post-Hoc Test Results (Dunn's Test with Bonferroni Correction) ---")

# Run Dunn's test on the same cleaned data
# We specify the data, the value column, the group column, and the p-value correction method
dunn_results = sp.posthoc_dunn(test_data_eth, 
                               val_col='fraction_poverty', 
                               group_col='ethnicity', 
                               p_adjust='bonferroni')

print(dunn_results)

print("\n--- How to Read This Table ---")
print("This table shows the p-value for the difference between each pair of groups.")
print("Any value in this table LESS THAN 0.05 is a *statistically significant* difference.")
# --- END POST-HOC TEST ---





# --- Analysis Plot 3: "Missingness" Analysis (Measurement Bias) ---
# This plot checks if SES data is "missing" more often for one group than another.
# We check for "null" values in the 'median_income' column (which we got from the LEFT JOIN).
print("Calculating missing SES data by race...")

# This part is for the PLOT
missing_data = df_analysis.groupby('race')['median_income'].apply(lambda x: x.isnull().mean() * 100)
missing_df = missing_data.reset_index(name='percent_missing_ses_data')
missing_df = missing_df.sort_values(by='percent_missing_ses_data', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=missing_df, y='race', x='percent_missing_ses_data')
plt.title('Percent of Participants with Missing SES Data by Race', fontsize=16)
plt.xlabel('Percent Missing SES Data (%)', fontsize=12)
plt.ylabel('Self-Reported Race', fontsize=12)

# --- BEGIN STATISTICAL TEST (CHI-SQUARE) ---
# We test if the proportion of missing data is independent of race.

# 1. Create a new column 'ses_data_missing' (True/False)
#    This makes creating the contingency table easy.
df_analysis['ses_data_missing'] = df_analysis['median_income'].isnull()

# 2. Create the contingency table (cross-tabulation)
#    This table shows the *raw counts* of (Missing vs. Not-Missing) vs. (Race).
contingency_table = pd.crosstab(df_analysis['race'], df_analysis['ses_data_missing'])

# 3. Run the Chi-Square test on the table
#    This returns the chi2 statistic, p-value, degrees of freedom, and expected frequencies
chi2, pvalue, dof, expected = stats.chi2_contingency(contingency_table)

# 4. Create the text to display on the plot
if pvalue < 0.05:
    stat_text = "Statistically Significant Difference\n"
    box_color = 'lightgreen'
    
    # Format the p-value string to handle very small numbers
    if pvalue < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {pvalue:.3f}"
    
    stat_text += f"(Chi-Square {p_text})"
    
else:
    box_color = 'lightgray'
    stat_text = f"No Significant Difference Found\n(Chi-Square p = {pvalue:.3f})"
    
# --- BEGIN TEXT BOX FIX ---
# 5. Add the text to the BOTTOM-RIGHT corner
plt.text(0.98, 0.02, stat_text,
         transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='bottom', horizontalalignment='right', # <-- CHANGED
         bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.5))
# --- END TEXT BOX FIX ---

# --- END STATISTICAL TEST ---

plt.show()
