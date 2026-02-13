import pandas as pd
import os

# 여기서 직접 경로 지정
FOLDER = "data"
FILENAME = "KS11.parquet"
FILE_PATH = os.path.join(FOLDER, FILENAME)

if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        exit(1)
    df = pd.read_parquet(FILE_PATH)
    print("Columns:")
    for col in df.columns:
        print(col)
