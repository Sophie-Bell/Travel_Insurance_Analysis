# --------------------------------------------------------------
# Necessary Libraries
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
from scipy.stats import ttest_ind,chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

# --------------------------------------------------------------
# Read the file
# --------------------------------------------------------------

with open("../data/processed/01_data_processed.pkl", "rb") as file:
    df = pickle.load(file)

# --------------------------------------------------------------
#  Collect Data
# --------------------------------------------------------------

chronic_insured = df[(df['TravelInsurance'] == 1) & (df['ChronicDiseases'] == 1)]
chronic_uninsured = df[(df['TravelInsurance'] == 0) & (df['ChronicDiseases'] == 1)]
nochronic_insured = df[(df['TravelInsurance'] == 1) & (df['ChronicDiseases'] == 0)]
nochronic_uninsured = df[(df['TravelInsurance'] == 0) & (df['ChronicDiseases'] == 0)]

print(f'Number of insured individuals with chronic diseases: {chronic_insured.shape[0]}')
print(f'Number of uninsured individuals with chronic diseases: {chronic_uninsured.shape[0]}')
print(f'Number of insured individuals without chronic diseases: {nochronic_insured.shape[0]}')
print(f'Number of uninsured individuals without chronic diseases: {nochronic_uninsured.shape[0]}')

# --------------------------------------------------------------
#  Evaluation
# --------------------------------------------------------------

# Chi-Squared Test for Independence
contingency_table = pd.crosstab(df['ChronicDiseases'], df['TravelInsurance'])
chi2, p, _, _ = chi2_contingency(contingency_table)

print("\nChi-Squared Test Results:")
print("Chi-Squared Statistic:", chi2)
print("p-value:", p)

alpha = 0.05  # Significance level
if p < alpha:
    print("Reject the null hypothesis: There is an association between chronic diseases and travel insurance.")
else:
    print("Fail to reject the null hypothesis: There is no association between chronic diseases and travel insurance.")