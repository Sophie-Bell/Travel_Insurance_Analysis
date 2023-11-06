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

non_governmentemployee = df[df['Employment Type'] == 'Private Sector']
governmentemployee = df[df['Employment Type'] == 'Government Sector']
income_nongovtemployee = non_governmentemployee['AnnualIncome']
income_govtemployee = governmentemployee['AnnualIncome']


print(f'Number of non government employees: {non_governmentemployee.shape[0]}')
print(f'Variance in annual income of non government employees: {np.var(income_nongovtemployee)}')
print(f'Number of government employees: {governmentemployee.shape[0]}')
print(f'Variance in annual income of government employees: {np.var(income_govtemployee )}')

# --------------------------------------------------------------
#  Visualize Income by Employment Type
# --------------------------------------------------------------

def visualize_income_by_employment(df, employment_column, income_column):
    # Visualize Annual Income by Employment Type
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=employment_column, y=income_column, hue=employment_column, palette="Set3", legend=False)
    plt.title('Annual Income by Employment Type')
    plt.xlabel('Employment Type')
    plt.ylabel('Annual Income')
    plt.show()

# --------------------------------------------------------------
#  Evaluation
# --------------------------------------------------------------

t_statistic, p_value = ttest_ind(income_govtemployee, income_nongovtemployee, equal_var=False)
print(f't_statistic: {t_statistic}\np_value: {p_value}')
print ("\ntwo-sample t-test p-value=", p_value)

alpha = 0.05  
if p_value < alpha:
    print("Reject the null hypothesis: Private Sector individuals have a significantly higher average annual income compared to those not in the private sector")
else:
    print("Fail to reject the null hypothesis: Private Sector individuals do not have a significantly higher average annual income compared to those not in the private sector")