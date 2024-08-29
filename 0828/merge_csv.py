import pandas as pd
from datetime import datetime, timedelta
import pickle

# Load GPS and IMU data
gps_df = pd.read_csv('07_ekf1.csv')
imu_df = pd.read_csv('imu1.csv')

# Convert the timestamps from nanoseconds to microseconds for easier handling
gps_df['%time'] = pd.to_datetime(gps_df['%time'], unit='ns')
imu_df['%time'] = pd.to_datetime(imu_df['%time'], unit='ns')

# Prepare lists to hold the final merged data
merged_data = []

# Initialize a variable to keep track of the last used GPS index
gps_index = 0

# Iterate through each IMU row
for imu_index, imu_row in imu_df.iterrows():
    imu_time = imu_row['%time']
    
    # Find the latest GPS timestamp that is less than the current IMU timestamp
    while gps_index < len(gps_df) and gps_df.at[gps_index, '%time'] <= imu_time:
        gps_index += 1
    
    # Use the last GPS entry that was valid
    gps_data = gps_df.iloc[gps_index - 1]
    
    # Extract the relevant fields
    row = {
        '%time': imu_time,
        'lat': gps_data['field.pos_x'],
        'lon': gps_data['field.pos_y'],
        'alt': gps_data['field.pos_z'],
        'roll': gps_data['field.ori_r'],
        'pitch': gps_data['field.ori_p'],
        'yaw': gps_data['field.ori_y'],
        'vn': gps_data['field.vel_x'],
        've': gps_data['field.vel_y'],
        'vu': gps_data['field.vel_z'],
        'ax': imu_row['field.linear_acceleration.x'],
        'ay': imu_row['field.linear_acceleration.y'],
        'az': imu_row['field.linear_acceleration.z'],
        'af': imu_row['field.linear_acceleration.x'],
        'al': imu_row['field.linear_acceleration.y'],
        'au': imu_row['field.linear_acceleration.z'],
        'wx': imu_row['field.angular_velocity.x'],
        'wy': imu_row['field.angular_velocity.y'],
        'wz': imu_row['field.angular_velocity.z'],
        'wf': imu_row['field.angular_velocity.x'],
        'wl': imu_row['field.angular_velocity.y'],
        'wu': imu_row['field.angular_velocity.z']
    }
    
    merged_data.append(row)

# Convert merged data to a DataFrame with the desired column order
column_order = [
    '%time', 'lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 
    'vn', 've', 'vu', 'ax', 'ay', 'az', 'af', 'al', 'au', 
    'wx', 'wy', 'wz', 'wf', 'wl', 'wu'
]

merged_df = pd.DataFrame(merged_data, columns=column_order)

# Save the merged data to a new CSV file
merged_df.to_csv('merged_output.csv', index=False)

print("Merged CSV file created successfully!")

# Save the DataFrame as a pickle file
with open('merged_output.p', 'wb') as f:
    pickle.dump(merged_df, f)

print("Merged CSV file created and saved as a pickle file successfully!")


# Load the pickle file
with open('merged_output.p', 'rb') as f:
    loaded_df = pickle.load(f)

# Display the first few rows
print(loaded_df.head())