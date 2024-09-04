import pickle
import pandas as pd
import os

# 파일 리스트를 정의합니다.
pickle_files = ['merged_output1.p', 'merged_output2.p', 'merged_output3.p']

# 각 파일에 대해 변환 작업을 수행합니다.
for pickle_file in pickle_files:
    # 1. dict 타입의 pickle 파일 로드
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # 2. 데이터 구조 확인
    print("Processing file:", pickle_file)
    print("Data type:", type(data))
    if isinstance(data, dict):
        for key, value in data.items():
            print("\nKey: {}".format(key))
            print("Type:", type(value))
            if isinstance(value, pd.Series):
                print("First 5 elements of Series:", value.head())
            elif isinstance(value, list):
                print("First 5 elements of list:", value[:5])
            elif isinstance(value, dict):
                print("Keys in sub-dict:", list(value.keys())[:5])
            elif isinstance(value, (int, float, str)):
                print("Value:", value)
            else:
                print("First 5 elements of {}: {}".format(type(value), str(value)[:100]))

    # 3. DataFrame으로 변환 가능한지 확인하고 변환
    try:
        df = pd.DataFrame(data)
        # 4. 변환된 DataFrame을 출력 (optional)
        print(df.head())

        # 파일 이름에서 확장자 제거
        base_file_name = os.path.splitext(pickle_file)[0]

        # 5. DataFrame을 CSV 파일로 저장
        csv_file_name = '{}_df.csv'.format(base_file_name)
        df.to_csv(csv_file_name, index=False)
        print("DataFrame saved as CSV:", csv_file_name)

        # 6. DataFrame을 새로운 pickle 파일로 저장
        pickle_file_name = '{}_df.p'.format(base_file_name)
        df.to_pickle(pickle_file_name)
        print("DataFrame saved as pickle:", pickle_file_name)

    except ValueError as e:
        print("Error converting data to DataFrame:", e)
