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

abroad_insured = df[(df['TravelInsurance'] == 1) & (df['EverTravelledAbroad'] == 1)]
abroad_uninsured = df[(df['TravelInsurance'] == 0) & (df['EverTravelledAbroad'] == 1)]
noabroad_insured = df[(df['TravelInsurance'] == 1) & (df['EverTravelledAbroad'] == 0)]
noabroad_uninsured = df[(df['TravelInsurance'] == 0) & (df['EverTravelledAbroad'] == 0)]

print(f'Number of insured individuals who have traveled abroad: {abroad_insured.shape[0]}')
print(f'Number of uninsured individuals who have traveled abroad: {abroad_uninsured.shape[0]}')
print(f'Number of insured individuals who have not traveled abroad: {noabroad_insured.shape[0]}')
print(f'Number of uninsured individuals who have not traveled abroad: {noabroad_uninsured.shape[0]}')

# --------------------------------------------------------------
#  Visualize the impact of travel insurance on traveling abroad
# --------------------------------------------------------------

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='EverTravelledAbroad', hue='TravelInsurance', palette="Set2")
plt.title('Impact of Travel Insurance on Traveling Abroad')
plt.xlabel('Ever Traveled Abroad')
plt.ylabel('Count')
plt.legend(title='Travel Insurance', labels=['Not Purchased', 'Purchased'])
plt.show()

# --------------------------------------------------------------
#  Evaluation
# --------------------------------------------------------------

contingency_table = pd.crosstab(df['EverTravelledAbroad'], df['TravelInsurance'])
chi2, p, _, _ = chi2_contingency(contingency_table)

print("\nChi-Squared Test Results:")
print("Chi-Squared Statistic:", chi2)
print("p-value:", p)

alpha = 0.05  # Significance level
if p < alpha:
    print("Reject the null hypothesis: There is an association between traveling abroad and travel insurance.")
else:
    print("Fail to reject the null hypothesis: There is no association between traveling abroad and travel insurance.")