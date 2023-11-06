# --------------------------------------------------------------
# Necessary Libraries
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats

# --------------------------------------------------------------
# Read the file
# --------------------------------------------------------------

with open("../data/processed/01_data_processed.pkl", "rb") as file:
    df = pickle.load(file)

# --------------------------------------------------------------
# Visualize a Numeric Column
# --------------------------------------------------------------

def visualize_numeric_column(df, column_name):
    # Create subplots as a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    plt.subplots_adjust(wspace=0.4)
    
    # Define different shades of blue
    shades_of_blue = ["#0072B5", "#5DA5E9", "#4169E1", "#1E90FF"]
    
    # Box Plot with blue color
    sns.boxplot(data=df, y=column_name, ax=axes[0, 0], color=shades_of_blue[0])
    axes[0, 0].set_title('Box Plot')
    
    # Violin Plot with a different shade of blue
    sns.violinplot(data=df, y=column_name, ax=axes[0, 1], color=shades_of_blue[1])
    axes[0, 1].set_title('Violin Plot')
    
    # Histogram with another shade of blue
    sns.histplot(data=df, x=column_name, ax=axes[1, 0], kde=True, color=shades_of_blue[2])
    axes[1, 0].set_title('Histogram')
    
    # Density Plot with a different shade of blue
    sns.kdeplot(data=df, x=column_name, ax=axes[1, 1], fill=True, color=shades_of_blue[3])
    axes[1, 1].set_title('Density Plot')
    
    # Set an overall title for the subplots
    fig.suptitle(f'Visualizations for {column_name} Feature')
    
    # Display the plots
    plt.show()

# --------------------------------------------------------------
# Calculate Distribution Statistics
# --------------------------------------------------------------

def calculate_distribution_stats(df, column_name, outlier_threshold=1.5):
    # Extract the specified column as a Series
    data = df[column_name]
    
    # Calculate mean and median
    mean = data.mean()
    median = data.median()
    
    # Calculate skewness
    skewness = data.skew()
    
    # Calculate interquartile range (IQR)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # Calculate lower and upper bounds for outliers
    lower_bound = Q1 - outlier_threshold * IQR
    upper_bound = Q3 + outlier_threshold * IQR
    
    # Identify outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    # Count the number of outliers
    num_outliers = len(outliers)
    
    return {
        "Mean": mean,
        "Median": median,
        "Skewness": skewness,
        "Total Outliers": num_outliers,
        "IQR": IQR
    }


# --------------------------------------------------------------
# Calculate and Visualize Correlation between 2 features
# --------------------------------------------------------------

def calculate_age_income_correlation(df, column1, column2):
    
    correlation = df[column1].corr(df[column2])
    
    # Visualize the correlation
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=column1, y=column2)
    plt.title(f'Correlation: {correlation:.2f}')
    plt.show()
    
    return correlation

# --------------------------------------------------------------
# Visualize Categorical and/or Binary Features
# --------------------------------------------------------------

def visualize_categorical_column(df, column_name):
    # Create subplots for categorical features
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Define different shades of blue
    shades_of_blue = ["#0072B5", "#5DA5E9"]
    
    # Count Plot for categories
    sns.countplot(data=df, x=column_name, hue=column_name, palette=shades_of_blue, ax=axes[0], legend=False)
    axes[0].set_title('Count Plot')
    
    # Pie Chart for categories
    data_counts = df[column_name].value_counts()
    axes[1].pie(data_counts, labels=data_counts.index, autopct='%1.1f%%', colors=shades_of_blue)
    axes[1].set_title('Pie Chart')
    
    # Set an overall title for the subplots
    fig.suptitle(f'Visualizations for {column_name}')
    
    # Display the plots
    plt.show()

# --------------------------------------------------------------
# Income by Employment Type
# --------------------------------------------------------------

def analyze_employment_and_insurance(df, employment_column, income_column, insurance_column):
    # 1. Visualize Annual Income by Employment Type
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=employment_column, y=income_column, hue=employment_column, palette="Set3", legend=False)
    plt.title('Annual Income by Employment Type')
    plt.xlabel('Employment Type')
    plt.ylabel('Annual Income')
    plt.show()

    # 2. Analyze Income Statistics
    income_statistics = df.groupby(employment_column)[income_column].describe()
    print("Income Statistics by Employment Type:")
    print(income_statistics)

    # 3. Visualize the Relationship with Travel Insurance
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=employment_column, hue=insurance_column, palette="Set2", legend=False)
    plt.title('Travel Insurance by Employment Type')
    plt.xlabel('Employment Type')
    plt.ylabel('Count')
    plt.show()

    # 4. Analyze Travel Insurance Statistics
    insurance_stats = df.groupby([employment_column, insurance_column]).size().unstack(fill_value=0)
    print("Travel Insurance Statistics by Employment Type:")
    print(insurance_stats)