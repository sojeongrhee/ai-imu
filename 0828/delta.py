import pickle
import csv
import torch

# Pickle 파일 로드
with open('../temp/delta_p.p', 'rb') as f:
    data = pickle.load(f)
    
if isinstance(data, dict):
    print("Data type: dict")
    print("Data keys: {}".format(list(data.keys())))  # dict의 키들을 출력

    # 각 키에 해당하는 값의 앞부분을 출력 (예: 앞의 5개 요소)
    for key, value in data.items():
        print("\nKey: {}".format(key))
        if isinstance(value, list) or isinstance(value, tuple):
            print("First 5 elements of {}: {}".format(key, value[:-1]))
        elif isinstance(value, dict):
            print("Keys of {}: {}".format(key, list(value.keys())[:-1]))
        elif isinstance(value, (int, float, str)):
            print("Value of {}: {}".format(key, value))
        elif isinstance(value, torch.Tensor):
            print("Tensor shape: {}, First 5 elements: {}".format(value.shape, value[:-1]))
        else:
            print("Type: {}, First 5 elements: {}".format(type(value), str(value)[:-1])) 

else:
    print("Data is of type: {}".format(type(data)))

# 데이터가 dict 타입인 경우 CSV로 저장
if isinstance(data, dict):
    with open('delta_p.csv', 'w', newline='') as csvfile:
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
                    row.append(value[i].item() if i < len(value) else "")
                else:
                    row.append(value if i == 0 else "")  # 단일 값인 경우 첫 번째 행에만 기록
            writer.writerow(row)

print("delta_p.p 파일이 delta_p.csv로 변환되었습니다.")
