# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set style for better looking plots
sns.set_style("whitegrid")

# Task 1: Load and Explore the Dataset
def load_and_explore_data():
    try:
        # Load the iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Display first few rows
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\n")
        
        # Explore structure
        print("Dataset info:")
        print(df.info())
        print("\n")
        
        # Check for missing values
        print("Missing values per column:")
        print(df.isnull().sum())
        print("\n")
        
        # Since there are no missing values in this dataset, we'll demonstrate cleaning anyway
        # In a real scenario with missing values, you might do:
        # df = df.dropna()  # to drop rows with missing values
        # or
        # df = df.fillna(method='ffill')  # to forward fill missing values
        
        return df
    
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

# Task 2: Basic Data Analysis
def perform_data_analysis(df):
    try:
        # Basic statistics
        print("Basic statistics for numerical columns:")
        print(df.describe())
        print("\n")
        
        # Group by species and compute mean
        print("Mean measurements by species:")
        print(df.groupby('species').mean())
        print("\n")
        
        # Additional interesting findings
        print("Additional observations:")
        print("1. Setosa has significantly smaller petal dimensions than other species.")
        print("2. Virginica has the largest measurements on average.")
        print("3. Versicolor is intermediate between setosa and virginica.")
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

# Task 3: Data Visualization
def create_visualizations(df):
    try:
        # Create a figure with subplots
        plt.figure(figsize=(15, 10))
        
        # 1. Line chart - showing trends (we'll use sepal length by index as a proxy for time)
        plt.subplot(2, 2, 1)
        df['sepal length (cm)'].plot(kind='line', color='green')
        plt.title('Sepal Length Trend (by index)')
        plt.xlabel('Index')
        plt.ylabel('Sepal Length (cm)')
        
        # 2. Bar chart - comparison of numerical value across categories
        plt.subplot(2, 2, 2)
        df.groupby('species')['petal length (cm)'].mean().plot(kind='bar', color=['blue', 'orange', 'green'])
        plt.title('Average Petal Length by Species')
        plt.xlabel('Species')
        plt.ylabel('Petal Length (cm)')
        
        # 3. Histogram - distribution of a numerical column
        plt.subplot(2, 2, 3)
        df['sepal width (cm)'].plot(kind='hist', bins=15, color='purple', edgecolor='black')
        plt.title('Distribution of Sepal Width')
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Frequency')
        
        # 4. Scatter plot - relationship between two numerical columns
        plt.subplot(2, 2, 4)
        colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
        for species, color in colors.items():
            subset = df[df['species'] == species]
            plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'], 
                        color=color, label=species, alpha=0.7)
        plt.title('Sepal Length vs Petal Length')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('iris_visualizations.png')  # Save the figure
        plt.show()
        
        # Additional visualization using seaborn
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='species', y='sepal width (cm)', data=df, palette='Set2')
        plt.title('Sepal Width Distribution by Species')
        plt.savefig('sepal_width_boxplot.png')
        plt.show()
        
    except Exception as e:
        print(f"An error occurred during visualization: {e}")

# Main execution
if __name__ == "__main__":
    print("Starting data analysis...\n")
    
    # Task 1
    iris_df = load_and_explore_data()
    
    if iris_df is not None:
        # Task 2
        perform_data_analysis(iris_df)
        
        # Task 3
        create_visualizations(iris_df)
        
    print("\nAnalysis complete!")