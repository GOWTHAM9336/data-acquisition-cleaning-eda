 
# Data Acquisition, Cleaning and EDA

## Project Description

This project performs data cleaning and exploratory data analysis (EDA) on a raw dataset using Python, Pandas, Matplotlib, and Seaborn. The goal is to prepare the dataset for future machine learning and predictive modeling.

## Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

## Tasks Performed

### 1. Data Loading

* Loaded the dataset using `pd.read_csv()`.
* Displayed the first five rows.
* Checked data types and dataset shape.

### 2. Missing Value Analysis

* Calculated missing values and percentages for all columns.
* Identified columns with more than 20% missing values.
* Filled missing values in numeric columns using the median.

### 3. Duplicate Removal

* Counted duplicate rows using `df.duplicated().sum()`.
* Removed duplicates using `df.drop_duplicates()`.

### 4. Data Type Conversion

* Converted incorrect numeric columns from object type to numeric.
* Converted repetitive text columns to category type.
* Compared memory usage before and after conversion.

### 5. Descriptive Statistics

* Generated summary statistics using `df.describe()`.
* Calculated skewness for all numeric columns.
* Identified the most skewed column.

### 6. Outlier Detection

* Used the IQR method to detect outliers.
* Counted outliers for selected numeric columns.
* Outliers were documented but not removed.

### 7. Visualizations

The following plots were created:

* Line Plot
* Bar Chart
* Histogram
* Scatter Plot
* Box Plot
* Correlation Heatmap

### 8. Correlation Analysis

* Calculated Pearson Correlation Matrix.
* Calculated Spearman Correlation Matrix.
* Compared Pearson and Spearman correlations.
* Identified variable pairs with the largest differences.

### 9. Grouped Aggregation

* Used `groupby()` with mean, standard deviation, and count.
* Identified groups with highest mean and highest variation.

### 10. Data Export

* Saved the cleaned dataset as `cleaned_data.csv`.

## Output Files

```text
dataset.csv
cleaned_data.csv
data_analysis.py
README.md
```

## Conclusion

The dataset was successfully cleaned, analyzed, and visualized. The final cleaned dataset is ready for further modeling and machine learning tasks.
