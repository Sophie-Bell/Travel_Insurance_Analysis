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

insured_family = df[df['TravelInsurance'] == 1]
uninsured_family = df[df['TravelInsurance'] == 0]
family_insured = insured_family['FamilyMembers']
family_uninsured = uninsured_family['FamilyMembers']

print(f'Number of insured individuals: {insured_family.shape[0]}')
print(f'Variance in the number of family members for insured individuals: {np.var(family_insured)}')
print(f'Number of uninsured individuals: {uninsured_family.shape[0]}')
print(f'Variance in the number of family members for uninsured individuals: {np.var(family_uninsured)}')

# --------------------------------------------------------------
#  Visualize Family Influence on Travel Insurance
# --------------------------------------------------------------

def visualize_family_influence(df, insurance_column, family_column):
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=insurance_column, y=family_column, hue=insurance_column, palette="Set3", legend=False)
    plt.title('Family Influence on Travel Insurance')
    plt.xlabel('Travel Insurance')
    plt.ylabel('Number of Family Members')
    plt.show()

# --------------------------------------------------------------
#  Evaluation
# --------------------------------------------------------------

t_statistic, p_value = ttest_ind(family_insured, family_uninsured, equal_var=False)
print(f't_statistic: {t_statistic}\np_value: {p_value}')
print("Two-sample t-test p-value =", p_value)

alpha = 0.05  # Significance level
if p_value < alpha:
    print("Reject the null hypothesis: There is an association between number of family members and travel insurance.")
else:
    print("Fail to reject the null hypothesis: There is no association between number of family members and travel insurance")