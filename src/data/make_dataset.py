# --------------------------------------------------------------
# Necessary Libraries
# --------------------------------------------------------------

import numpy as np
import pandas as pd

# --------------------------------------------------------------
# Read the file
# --------------------------------------------------------------

df = pd.read_csv('../data/raw/TravelInsurancePrediction.csv')

# --------------------------------------------------------------
# Drop 'Unnamed 0' Column
# --------------------------------------------------------------

df.drop("Unnamed: 0", axis=1, inplace=True)

# --------------------------------------------------------------
# Convert Binary Columns
# --------------------------------------------------------------

def convert_yes_no_to_binary(df, columns):
    for column in columns:
        df[column] = df[column].map({'Yes': 1, 'No': 0})
    return df

# --------------------------------------------------------------
# Change Employment Type Values
# --------------------------------------------------------------

df['Employment Type'] = df['Employment Type'].replace('Private Sector/Self Employed', 'Private Sector')

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df.to_pickle("../data/processed/01_data_processed.pkl")