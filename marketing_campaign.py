import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\prash\OneDrive\Documents\PERSONAL LIBRARY\Programming\GitHub\marketing_campaign_dataset\marketing_campaign_dataset.csv")
print(df.columns)
print(df['Campaign_ID'].value_counts().unique().sum())
print(df['Campaign_ID'].value_counts().sum())

##First Five Rows
print("First Five Rows")
print(df.head())

##Last Five Rows
print("Last Five Rows")
print(df.tail())

##Random samples
print("Random Sample")
print(df.sample(5, random_state=645))

##Dataset Info
print("Dataset Information")
print(df.info())

##Data Types
print("Data Types")
print(df.dtypes)

##Null Values in the Dataset
print("Null values in the datset")
print(df.isnull().sum())

##Tolist
print(df.columns.tolist())

##Statistical Summary
print("Statistical Summary:")
print(df.describe(include='all'))

##Missing Values
print("Missnig Values :")
print(df.isnull().sum())

##Duplicated Values
print("Duplicated")
print(df.duplicated().sum())

df_clean=df.copy()

print("Original Shape:", df.shape)
print("Working Dataset Shape:", df_clean.shape)

print(df_clean.dtypes)

##Cleaning 
df_clean["Duration"] = (df_clean["Duration"].str.replace(" days", "", regex=False).astype(int))

print(df_clean["Duration"].head())
print(df_clean["Duration"].dtype)

df_clean["Acquisition_Cost"] = (df_clean["Acquisition_Cost"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float))

print(df_clean["Acquisition_Cost"].head())
print(df_clean["Acquisition_Cost"].dtype)

df_clean["Date"] = pd.to_datetime(df_clean["Date"])
print(df_clean["Date"].head())
print(df_clean["Date"].dtype)

##Data Types
print(df_clean.dtypes)

##Rechecking_Null values
print("Missing Values")
print(df_clean.isnull().sum())

print("Total Missing Values:",df_clean.isnull().sum())


## Duplicated Values
print("Duplicated Rows :",df_clean.duplicated().sum())

##chart for Campaign_Type
plt.figure(figsize=(10, 6)) 

## Seting the size of the chart
sns.countplot(data=df_clean, x='Campaign_Type', palette='viridis', 
              order=df_clean['Campaign_Type'].value_counts().index)

plt.title('Total Count of Each Campaign Type', fontsize=14)
plt.xlabel('Campaign Type', fontsize=12)
plt.ylabel('Number of Campaigns', fontsize=12)
plt.show()

##[our campaign types are evenly distributed]
# Influencer: 40,169
# Search: 40,157
# Display: 39,987
# Email: 39,870
# Social Media: 39,817

##Bivariate Analysis

avg_roi_by_channel = df_clean.groupby('Channel_Used')['ROI'].mean().reset_index()
print("Average ROI by Channel:")
print(avg_roi_by_channel)

# Visualize the Average ROI
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_roi_by_channel, x='Channel_Used', y='ROI', palette='magma')

plt.title('Average ROI by Marketing Channel', fontsize=14)
plt.xlabel('Channel Used', fontsize=12)
plt.ylabel('Average ROI', fontsize=12)
plt.show()

##Marketing = ChannelAverage (ROIFacebook-5.02,Website-5.01,Email-5.00,Google Ads-5.00,Instagram-4.99,YouTube-4.99)

## average Conversion Rate by Target Audience
avg_conv_by_aud = df_clean.groupby('Target_Audience')['Conversion_Rate'].mean().reset_index()
print("Average Conversion Rate by Audience:")
print(avg_conv_by_aud)
###Average Conversion Rate by Audience:
### Target_Audience  Conversion_Rate
#      All_Ages           0.079975
#    Men 18-24            0.080240
#    Men 25-34            0.080132
#    Women 25-34          0.079899
#    Women 35-44          0.080102


##Average_conversion_rate_by_target_audience
avg_conv_by_aud = df_clean.groupby('Target_Audience')['Conversion_Rate'].mean().reset_index()
print("Average Conversion Rate by Audience:")
print(avg_conv_by_aud)

# Visualize with a Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_conv_by_aud, x='Target_Audience', y='Conversion_Rate', palette='coolwarm')

plt.title('Average Conversion Rate by Target Audience', fontsize=14)
plt.xlabel('Target Audience', fontsize=12)
plt.ylabel('Average Conversion Rate', fontsize=12)
plt.show()

# Calculating Acquisition Cost by Duration
avg_cost_by_dur = df_clean.groupby('Duration')['Acquisition_Cost'].mean().reset_index()
print("Average Cost by Duration:")
print(avg_cost_by_dur)

# Visualize with a Line Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_cost_by_dur, x='Duration', y='Acquisition_Cost', marker='o', color='b', linewidth=2)

plt.title('Average Acquisition Cost by Campaign Duration', fontsize=14)
plt.xlabel('Duration (in days)', fontsize=12)
plt.ylabel('Average Acquisition Cost ($)', fontsize=12)
plt.ylim(12000, 13000) # Setting a tight y-limit to zoom in on the numbers
plt.show()

##average cost based on the campaign's length:
# 15 days: $12,507.60
# 30 days: $12,490.21
# 45 days: $12,505.10
# 60 days: $12,514.78

###Multivariate Analysis
numerical_cols = df_clean.select_dtypes(include=['int32', 'int64', 'float64']).columns
df_numeric = df_clean[numerical_cols]

###correlation matrix
corr_matrix = df_numeric.corr()

###Heatmap
plt.figure(figsize=(12, 8))
# annot=True puts the actual numbers in the squares, cmap chooses the color scheme
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title('Correlation Heatmap of Marketing Campaign Variables', fontsize=16)
plt.show()