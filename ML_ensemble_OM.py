import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


# Path to your Excel file
file_path = r"C:/Users/Dell/Downloads/Modified_200hz_Cycling_Data.xlsx"

# Read the sheet into a DataFrame
OM_df = pd.read_excel(file_path, sheet_name=1)  # Indexing is zero-based

# Display the DataFrame
print("OM DataFrame:")
#print(OM_df.head())

# Step 1: Add a column for the original index
OM_df['Original Index'] = OM_df.index + 2

# Step 2: Group by all columns except 'OM Run Time', 'OM Failure', and 'Original Index', 
# and find the row with the maximum 'OM Run Time' within each group
grouping_columns = OM_df.columns.difference(['OM Run Time', 'OM Failure', 'Original Index']).tolist()
grouped_df = OM_df.loc[OM_df.groupby(grouping_columns)['OM Run Time'].idxmax()]

# Step 3: Reset index to get a clean DataFrame with the original index
grouped_df = grouped_df.reset_index(drop=True)

# Step 4: Sort the DataFrame by 'Original Index' in increasing order
grouped_df = grouped_df.sort_values(by='Original Index')

# Step 5: Drop the default integer index column and set 'Original Index' as the index
grouped_df = grouped_df.set_index('Original Index')

# Step 6: Save the grouped DataFrame to an Excel file
output_file_path = r"C:/Users/Dell/Downloads/Grouped_OM_Data.xlsx"
grouped_df.to_excel(output_file_path, index=True)

# Display the grouped DataFrame with the original index
print(grouped_df)

# Define the mappings
type_mapping = {1: 'Raw 6061', 2: 'Hard coat 7075', 3: 'Cepton TiNi', 4: 'Standard Cepton', 5: 'Koito'}
om_kapton_mapping = {0: 'No Kapton', 1: 'Top Kapton', 2: 'Bottom Kapton', 3: 'Top & Bottom Kapton'}
om_washer_mapping = {1: 'SS Top Washer', 2: 'Zinc Top Washer', 3: 'Brass Washer', 4: 'Al Top Washer'}
om_failure_mapping = {0: 'No', 1: 'Yes'}

# Apply the mappings to the corresponding columns
grouped_df['Type'] = grouped_df['Type'].map(type_mapping)
grouped_df['OM Kapton'] = grouped_df['OM Kapton'].map(om_kapton_mapping)
grouped_df['OM Washers'] = grouped_df['OM Washers'].map(om_washer_mapping)
grouped_df['OM Failure'] = grouped_df['OM Failure'].map(om_failure_mapping)

# Define the path for the new Excel file
output_file_path = r"C:/Users/Dell/Downloads/Transformed_Grouped_OM_Data.xlsx"

# Save the transformed DataFrame to a new Excel file
grouped_df.to_excel(output_file_path, index=True)

# Display the transformed DataFrame
print("Transformed Grouped OM DataFrame:")
print(grouped_df.head())

# Define features and target variable
features = ['Type', 'OM Kapton', 'OM Washers']
target = 'OM Run Time'

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(grouped_df[features])
y = grouped_df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestRegressor
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Print feature importance
print("Feature Importance:")
print(feature_importance)

# Optionally, predict on test set and evaluate the model performance
# y_pred = model.predict(X_test)
# Evaluate model performance metrics if needed

# Extract top N important features
top_n = 3  # Number of top features to display
top_features = feature_importance.head(top_n).index.tolist()

# Print insights on top features
print("\nTop {} features for predicting high OM Run Time:".format(top_n))
for feature in top_features:
    print("- {}: {:.2f}%".format(feature, feature_importance[feature] * 100))

# Optionally, you can also analyze combinations of factors if they were one-hot encoded
# For example, if 'Type' was encoded into multiple columns:
# type_columns = [col for col in X.columns if col.startswith('Type_')]
# type_importance = feature_importance[type_columns].sum()

# Print insights on combinations if applicable
# print("\nCombinations importance:")
# print(type_importance)