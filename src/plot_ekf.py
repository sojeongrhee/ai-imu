import os
import pandas as pd
import matplotlib.pyplot as plt

# csv 파일 로드
data_path = "../dataset/sheco_data"
exp_num = 2 # 실험 번호에 따라 변경 가능
# df = pd.read_csv(f"{data_path}/ekf_py_{exp_num}.csv")
df = pd.read_csv(f"{data_path}/ekf_output{exp_num}.csv")

# 시간값
time = df['%time'] / 1e9  # 시간 단위가 나노초로 되어 있을 경우

# 저장할 경로 설정
save_path = f"{data_path}/plots"
os.makedirs(save_path, exist_ok=True)

# Position 그래프
plt.figure(figsize=(10, 6))
plt.plot(time, df['field.pos_x'], label='Position X')
plt.plot(time, df['field.pos_y'], label='Position Y')
plt.plot(time, df['field.pos_z'], label='Position Z')
plt.title('Position Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.savefig(f"{save_path}/position_plot_exp_{exp_num}.png")  # 이미지 저장
plt.show()

# Velocity 그래프
plt.figure(figsize=(10, 6))
plt.plot(time, df['field.vel_x'], label='Velocity X')
plt.plot(time, df['field.vel_y'], label='Velocity Y')
plt.plot(time, df['field.vel_z'], label='Velocity Z')
plt.title('Velocity Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.savefig(f"{save_path}/velocity_plot_exp_{exp_num}.png")  # 이미지 저장
plt.show()

# Orientation 그래프 (Roll, Pitch, Yaw)
plt.figure(figsize=(10, 6))
plt.plot(time, df['field.ori_r'], label='Roll')
plt.plot(time, df['field.ori_p'], label='Pitch')
plt.plot(time, df['field.ori_y'], label='Yaw')
plt.title('Orientation (Roll, Pitch, Yaw) Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Orientation (rad)')
plt.legend()
plt.grid(True)
plt.savefig(f"{save_path}/orientation_plot_exp_{exp_num}.png")  # 이미지 저장
plt.show()
