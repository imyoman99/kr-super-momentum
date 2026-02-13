import pandas as pd
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print("사용법: python scripts/compare_asset_lists.py <파일1> <파일2>")
    sys.exit(1)

file1, file2 = sys.argv[1], sys.argv[2]

# 파일 경로 보정
file1 = str(Path(file1).resolve())
file2 = str(Path(file2).resolve())

print(f"[1] {file1}")
print(f"[2] {file2}")

# CSV 읽기
try:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
except Exception as e:
    print(f"CSV 읽기 오류: {e}")
    sys.exit(2)

# 행/열/날짜 비교
print(f"행 개수: {len(df1)} vs {len(df2)}")
print(f"열 개수: {len(df1.columns)} vs {len(df2.columns)}")

# 날짜 컬럼 자동 탐색
date_col = [c for c in df1.columns if 'DATE' in c or '날짜' in c]
if date_col:
    date_col = date_col[0]
    dates1 = set(df1[date_col])
    dates2 = set(df2[date_col])
    only1 = sorted(list(dates1 - dates2))
    only2 = sorted(list(dates2 - dates1))
    print(f"파일1에만 있는 날짜: {only1[:5]} ... (총 {len(only1)}개)")
    print(f"파일2에만 있는 날짜: {only2[:5]} ... (총 {len(only2)}개)")
else:
    print("날짜 컬럼을 찾을 수 없습니다.")

# 자산총액 비교
if '자산총액' in df1.columns and '자산총액' in df2.columns:
    diff = (df1['자산총액'] != df2['자산총액'])
    diff_count = diff.sum()
    print(f"자산총액이 다른 날짜 수: {diff_count}")
    if diff_count > 0:
        print(df1.loc[diff, ['DATE','자산총액']].head())
        print(df2.loc[diff, ['DATE','자산총액']].head())
else:
    print("자산총액 컬럼을 찾을 수 없습니다.")

# 매매내역 비교(티커)
ticker_cols = [c for c in df1.columns if '티커' in c]
if ticker_cols:
    for col in ticker_cols:
        diff = (df1[col] != df2[col])
        diff_count = diff.sum()
        print(f"{col} 값이 다른 날짜 수: {diff_count}")
else:
    print("티커 관련 컬럼을 찾을 수 없습니다.")

print("비교 완료.")
