import pandas as pd
from datetime import datetime
import pickle

def read_csv(filepath):
    return pd.read_csv(filepath)

def synchronize_and_merge(ekf_data, imu_data, output_filepath):
    merged_data = []
    imu_index = 0
    imu_timestamp_ns = int(imu_data.iloc[imu_index]['%time'])

    for _, ekf_row in ekf_data.iterrows():
        ekf_timestamp_ns = int(ekf_row['%time'])

        # 조건 1: EKF의 timestamp가 IMU의 timestamp보다 나중이어야 함
        while imu_index < len(imu_data) - 1 and int(imu_data.iloc[imu_index + 1]['%time']) < ekf_timestamp_ns:
            imu_index += 1
            imu_timestamp_ns = int(imu_data.iloc[imu_index]['%time'])

        # IMU와 EKF의 가장 가까운 timestamp를 찾아서 데이터를 합침
        if imu_timestamp_ns <= ekf_timestamp_ns:
            imu_row = imu_data.iloc[imu_index]

            merged_row = {
                '%time': datetime.fromtimestamp(ekf_timestamp_ns / 1e9).isoformat(),
                'lat': ekf_row['field.pos_x'],
                'lon': ekf_row['field.pos_y'],
                'alt': ekf_row['field.pos_z'],
                'roll': ekf_row['field.ori_r'],
                'pitch': ekf_row['field.ori_p'],
                'yaw': ekf_row['field.ori_y'],
                'vn': ekf_row['field.vel_x'],
                've': ekf_row['field.vel_y'],
                'vu': ekf_row['field.vel_z'],
                'ax': imu_row['field.linear_acceleration.x'],
                'ay': imu_row['field.linear_acceleration.y'],
                'az': imu_row['field.linear_acceleration.z'],
                'af': imu_row['field.linear_acceleration.x'],  # Duplicates as per instruction
                'al': imu_row['field.linear_acceleration.y'],  # Duplicates as per instruction
                'au': imu_row['field.linear_acceleration.z'],  # Duplicates as per instruction
                'wx': imu_row['field.angular_velocity.x'],
                'wy': imu_row['field.angular_velocity.y'],
                'wz': imu_row['field.angular_velocity.z'],
                'wf': imu_row['field.angular_velocity.x'],  # Duplicates as per instruction
                'wl': imu_row['field.angular_velocity.y'],  # Duplicates as per instruction
                'wu': imu_row['field.angular_velocity.z'],  # Duplicates as per instruction
            }

            merged_data.append(merged_row)

    # 열 순서 지정
    column_order = [
        '%time', 'lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 
        'vn', 've', 'vu', 'ax', 'ay', 'az', 'af', 'al', 'au', 
        'wx', 'wy', 'wz', 'wf', 'wl', 'wu'
    ]

    # Pandas DataFrame으로 변환 및 열 순서 지정
    merged_df = pd.DataFrame(merged_data, columns=column_order)

    # CSV 파일로 저장
    merged_df.to_csv(output_filepath, index=False)

    return merged_df

# ekf.csv와 imu.csv 파일을 읽어오기
ekf_data = read_csv('07_ekf1.csv')
imu_data = read_csv('imu1.csv')

# 합친 데이터를 저장할 파일 경로
output_filepath = 'merged_output_0830.csv'

# 동기화 및 병합 작업 실행
merged_df = synchronize_and_merge(ekf_data, imu_data, output_filepath)

# Save the DataFrame as a pickle file
with open('merged_output_0830.p', 'wb') as f:
    pickle.dump(merged_df, f)

print("Merged CSV file created and saved as a pickle file successfully!")

# Load the pickle file
with open('merged_output_0830.p', 'rb') as f:
    loaded_df = pickle.load(f)

# Display the first few rows
print(loaded_df.head())
