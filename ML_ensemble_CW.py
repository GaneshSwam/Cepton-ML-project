import pandas as pd
import numpy as np  # Add this line to import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Path to your Excel file
file_path = r"C:/Users/Dell/Downloads/Modified_200hz_Cycling_Data.xlsx"

# Read the sheet into a DataFrame
CW_df = pd.read_excel(file_path, sheet_name=2)  # Indexing is zero-based

# Step 1: Drop the default integer index column (resetting the index)
CW_df = CW_df.reset_index(drop=True)

# Define the mappings
type_mapping = {1: 'Raw', 2: 'Ni-plated'}
cw_kapton_mapping = {0: 'No Kapton', 1: 'Top Kapton', 2: 'Bottom Kapton', 3: 'Bottom + Top Kapton'}
cw_washer_mapping = {1: 'SS Top Washer', 2: 'Zinc Top Washer', 3: 'Brass Washer', 4: 'Al Top Washer', 5: 'Zinc Top & Bottom Washer'}
cw_failure_mapping = {0: 'No', 1: 'Yes'}

# Apply the mappings to the corresponding columns
CW_df['Type'] = CW_df['Type'].map(type_mapping)
CW_df['CW Kapton'] = CW_df['CW Kapton'].map(cw_kapton_mapping)
CW_df['CW Washers'] = CW_df['CW Washers'].map(cw_washer_mapping)
CW_df['CW Failure'] = CW_df['CW Failure'].map(cw_failure_mapping)

# Define the path for the new Excel file
output_file_path = r"C:/Users/Dell/Downloads/Transformed_Grouped_CW_Data.xlsx"

# Save the transformed DataFrame to a new Excel file, overwriting if it exists
CW_df.to_excel(output_file_path, index=False)

# Step 2: Group by unique configurations and aggregate features
grouped_df = CW_df.groupby(['Type', 'X(1239)', 'Y(1197)', 'Freq X', 'Freq Y', 'CW Kapton', 'CW Washers', 'CW Torque', 'CW T0', 'CW T1', 'CW T2', 'CW T3', 'CW T4']).agg({
    'CW Run Time': 'max',  # Assuming we take the maximum run time as the predicted failure time
}).reset_index()

# Step 3: Define features and target
X = grouped_df.drop(columns=['CW Run Time'])  # Dropping 'CW Run Time' from features
y = grouped_df['CW Run Time']

# Step 4: Define preprocessing steps for categorical and numerical features
categorical_features = ['Type', 'CW Kapton', 'CW Washers']
numeric_features = ['X(1239)', 'Y(1197)', 'Freq X', 'Freq Y', 'CW Torque', 'CW T0', 'CW T1', 'CW T2', 'CW T3', 'CW T4']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)  # 'passthrough' keeps numeric features as-is
    ])

# Step 5: Define the model (Random Forest Regressor)
model = RandomForestRegressor(random_state=42)

# Step 6: Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

# Step 7: Train the model
pipeline.fit(X, y)

# Step 8: Predict run time until failure for each unique configuration
predicted_runtimes = pipeline.predict(X)

# Step 9: Adjust predicted run times to ensure they do not exceed the maximum observed run time for each configuration
grouped_df['Predicted_Run_Time_Until_Failure'] = np.minimum(predicted_runtimes, grouped_df['CW Run Time'])
print(grouped_df[['Type', 'X(1239)', 'Y(1197)', 'Freq X', 'Freq Y', 'CW Kapton', 'CW Washers', 'CW Torque', 'CW T0', 'CW T1', 'CW T2', 'CW T3', 'CW T4', 'Predicted_Run_Time_Until_Failure']])
