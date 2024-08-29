import pickle
import csv
import torch

# Pickle 파일 로드
with open('../data/merged_output_2.p', 'rb') as f:
    data = pickle.load(f)

# 데이터가 dict 타입인 경우 CSV로 저장
if isinstance(data, dict):
    with open('merged_output_2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 헤더 작성 (dict의 키들)
        writer.writerow(data.keys())

        # 가장 큰 길이 찾기 (모든 값들이 동일한 길이를 가지도록 맞추기 위함)
        max_len = max(len(value) if isinstance(value, (list, tuple, torch.Tensor)) else 1 for value in data.values())

        # 값 작성
        for i in range(max_len):
            row = []
            for key in data.keys():
                value = data[key]
                if isinstance(value, (list, tuple)):
                    row.append(value[i] if i < len(value) else "")
                elif isinstance(value, torch.Tensor):
                    if len(value.shape) > 1:
                        row.append(value[i].numpy().tolist() if i < len(value) else "")
                    else:
                        row.append(value[i].item() if i < len(value) else "")
                else:
                    row.append(value if i == 0 else "")  # 단일 값인 경우 첫 번째 행에만 기록
            writer.writerow(row)

print("done")
