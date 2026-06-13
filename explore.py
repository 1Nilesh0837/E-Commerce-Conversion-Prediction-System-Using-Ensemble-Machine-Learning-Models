import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
public_test = pd.read_csv('public_test.csv')
private_test = pd.read_csv('private_test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print("=== SHAPES ===")
print(f"Train: {train.shape}")
print(f"Public Test: {public_test.shape}")
print(f"Private Test: {private_test.shape}")
print(f"Sample Sub: {sample_sub.shape}")

print("\n=== COLUMNS ===")
print(train.columns.tolist())

print("\n=== DTYPES ===")
print(train.dtypes)

print("\n=== HEAD ===")
print(train.head(5).to_string())

print("\n=== DESCRIBE ===")
print(train.describe().to_string())

print("\n=== MISSING VALUES ===")
print(train.isnull().sum())

print("\n=== TARGET DISTRIBUTION ===")
print(train['Converted'].value_counts())
print(train['Converted'].value_counts(normalize=True))

print("\n=== CATEGORICALS ===")
for c in train.select_dtypes('object').columns:
    print(f"\n{c}: {train[c].nunique()} unique")
    print(train[c].value_counts().head(10))

print("\n=== CONVERSION BY DEVICE ===")
print(pd.crosstab(train['Device_Type'], train['Converted'], normalize='index'))

print("\n=== CONVERSION BY TRAFFIC SOURCE ===")
print(pd.crosstab(train['Traffic_Source'], train['Converted'], normalize='index'))

print("\n=== SAMPLE SUBMISSION ===")
print(sample_sub.head())
print(sample_sub.columns.tolist())

print("\n=== PUBLIC TEST LABELS? ===")
print('Converted' in public_test.columns)
if 'Converted' in public_test.columns:
    print(public_test['Converted'].value_counts())

print("\n=== PRIVATE TEST COLUMNS ===")
print(private_test.columns.tolist())

# Check Campaign_Code patterns
print("\n=== CAMPAIGN CODE CONVERSION RATES ===")
camp_conv = train.groupby('Campaign_Code')['Converted'].agg(['mean','count']).sort_values('mean', ascending=False)
print(camp_conv.head(20))

print("\n=== BROWSER VERSION CONVERSION RATES ===")
bv_conv = train.groupby('Browser_Version')['Converted'].agg(['mean','count']).sort_values('mean', ascending=False)
print(bv_conv.head(20))

print("\n=== CITY TIER ===")
if 'City_Tier' in train.columns:
    print(pd.crosstab(train['City_Tier'], train['Converted'], normalize='index'))

print("\n=== NUMERICAL CORRELATIONS WITH TARGET ===")
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if 'Converted' in num_cols:
    corrs = train[num_cols].corr()['Converted'].sort_values(ascending=False)
    print(corrs)
