import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATASET


df = pd.read_csv("your_dataset.csv")

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATA TYPES")
print(df.dtypes)

print("\nSHAPE")
print(df.shape)


# 2. NULL VALUE ANALYSIS


null_count = df.isnull().sum()
null_percent = (df.isnull().sum() / df.shape[0]) * 100

null_table = pd.DataFrame({
    "Null Count": null_count,
    "Null Percentage": null_percent
})

print("\nNULL VALUE ANALYSIS")
print(null_table)

high_null_cols = null_percent[null_percent > 20]

print("\nColumns exceeding 20% null rate:")
print(high_null_cols)

# Fill numeric columns with median if null rate <20%

for col in df.columns:
    if df[col].isnull().sum() > 0:

        if pd.api.types.is_numeric_dtype(df[col]):

            if null_percent[col] < 20:
                df[col].fillna(df[col].median(), inplace=True)


# 3. DUPLICATE DETECTION


duplicates_before = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates_before)

null_before = (df.isnull().sum()/len(df))*100

df = df.drop_duplicates()

duplicates_after = df.duplicated().sum()

rows_removed = duplicates_before

print("Rows Removed:", rows_removed)

null_after = (df.isnull().sum()/len(df))*100

comparison = pd.DataFrame({
    "Before": null_before,
    "After": null_after
})

print("\nNull Percentage Change")
print(comparison)


# 4. DATA TYPE CORRECTION

memory_before = df.memory_usage(deep=True).sum()

print("\nMemory Before:", memory_before)

# Example object → numeric conversion

for col in df.columns:

    if df[col].dtype == "object":

        try:
            converted = pd.to_numeric(df[col], errors='coerce')

            if converted.notnull().sum() > 0:
                df[col] = converted
                print(f"{col} converted to numeric")
                break

        except:
            pass

# Convert repetitive object column to category

for col in df.columns:

    if df[col].dtype == "object":

        if df[col].nunique() < len(df)*0.5:
            df[col] = df[col].astype('category')
            print(f"{col} converted to category")
            break

memory_after = df.memory_usage(deep=True).sum()

print("Memory After:", memory_after)


# 5. DESCRIPTIVE STATS + SKEWNESS

numeric_cols = df.select_dtypes(include=np.number).columns

print("\nDESCRIPTIVE STATISTICS")
print(df[numeric_cols].describe())

skewness = {}

for col in numeric_cols:
    skewness[col] = df[col].skew()

skew_df = pd.DataFrame.from_dict(
    skewness,
    orient='index',
    columns=['Skewness']
)

print("\nSKEWNESS")
print(skew_df)

highest_skew_col = max(
    skewness,
    key=lambda x: abs(skewness[x])
)

print("\nHighest Absolute Skewness Column:")
print(highest_skew_col)

# 6. IQR OUTLIER ANALYSIS


print("\nOUTLIER ANALYSIS")

outlier_columns = numeric_cols[:2]

for col in outlier_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(f"\nColumn: {col}")
    print("Outliers:", len(outliers))


# 7. VISUALIZATIONS


# Line Plot

plt.figure(figsize=(8,5))
plt.plot(df[numeric_cols[0]])
plt.title("Line Plot")
plt.xlabel("Index")
plt.ylabel(numeric_cols[0])
plt.show()

# Bar Chart

cat_cols = df.select_dtypes(
    include=['object','category']
).columns

if len(cat_cols) > 0:

    grp = df.groupby(cat_cols[0])[numeric_cols[0]].mean()

    plt.figure(figsize=(8,5))
    grp.plot(kind='bar')
    plt.title("Mean by Category")
    plt.xlabel(cat_cols[0])
    plt.ylabel("Mean")
    plt.show()

# Histogram

plt.figure(figsize=(8,5))
sns.histplot(df[highest_skew_col], bins=20)
plt.title("Histogram")
plt.show()

# Scatter Plot

if len(numeric_cols) >= 2:

    plt.figure(figsize=(8,5))
    sns.scatterplot(
        x=df[numeric_cols[0]],
        y=df[numeric_cols[1]]
    )
    plt.title("Scatter Plot")
    plt.show()

# Box Plot

if len(cat_cols) > 0:

    plt.figure(figsize=(8,5))
    sns.boxplot(
        x=cat_cols[0],
        y=numeric_cols[0],
        data=df
    )
    plt.title("Box Plot")
    plt.show()


# 8. PEARSON CORRELATION HEATMAP


corr_matrix = df[numeric_cols].corr()

print("\nPEARSON CORRELATION")
print(corr_matrix)

plt.figure(figsize=(10,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# Highest correlation pair

corr_unstack = corr_matrix.abs().unstack()

corr_unstack = corr_unstack[
    corr_unstack < 1
]

highest_pair = corr_unstack.idxmax()

print("\nHighest Correlation Pair:")
print(highest_pair)

# 9A. IMPUTATION STRATEGY COMPARISON

abs_skew = skew_df["Skewness"].abs().sort_values(
    ascending=False
)

top2 = abs_skew.index[:2]

print("\nTOP 2 SKEWED COLUMNS")

for col in top2:

    mean_val = df[col].mean()
    median_val = df[col].median()

    print(
        f"{col} -> Mean={mean_val:.2f}, "
        f"Median={median_val:.2f}"
    )

    df[col].fillna(median_val, inplace=True)

print("\nRemaining Nulls:")
print(df[top2].isnull().sum())

# 9B. SPEARMAN CORRELATION

spearman = df[numeric_cols].corr(
    method='spearman'
)

print("\nSPEARMAN")
print(spearman)

pairs = []

for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):

        c1 = numeric_cols[i]
        c2 = numeric_cols[j]

        pear = corr_matrix.loc[c1,c2]
        spear = spearman.loc[c1,c2]

        diff = abs(spear - pear)

        pairs.append([
            c1,
            c2,
            pear,
            spear,
            diff
        ])

diff_df = pd.DataFrame(
    pairs,
    columns=[
        "Column1",
        "Column2",
        "Pearson",
        "Spearman",
        "Difference"
    ]
)

diff_df = diff_df.sort_values(
    "Difference",
    ascending=False
)

print("\nTOP 3 DIFFERENCES")
print(diff_df.head(3))

# 9C. GROUPED AGGREGATION

if len(cat_cols) > 0:

    agg = df.groupby(
        cat_cols[0]
    )[numeric_cols[0]].agg(
        ['mean','std','count']
    )

    print("\nGROUPED AGGREGATION")
    print(agg)

    highest_mean = agg['mean'].idxmax()
    highest_std = agg['std'].idxmax()

    print("\nHighest Mean Group:")
    print(highest_mean)

    print("\nHighest Std Group:")
    print(highest_std)

    ratio = (
        agg['mean'].max() /
        agg['mean'].min()
    )

    print("\nMean Ratio:", ratio)

# 10. SAVE CLEAN DATA

df.to_csv(
    "cleaned_data.csv",
    index=False
)

print(
    "\ncleaned_data.csv saved successfully!"
)
