import pandas as pd
import numpy as np

# 1. 설정
start_date = "2023-01-01"
end_date = "2025-12-31"
initial_capital = 10000000  # 초기 자본 1,000만원
dates = pd.date_range(start=start_date, end=end_date, freq='B') # 영업일 기준

# 2. 가상 수익률 생성 (정규분포 + 하락 구간 추가)
np.random.seed(42) # 결과 재현을 위해 시드 고정
daily_returns = np.random.normal(0.0006, 0.012, len(dates)) # 평균 0.06%, 변동성 1.2%

# 특정 구간에 하락장 시뮬레이션 (예: 200~250일차에 큰 하락)
daily_returns[200:250] -= 0.02 

# 3. 자산 가치(Equity) 계산
# 누적 수익률 계산: (1 + r1) * (1 + r2) ...
equity_curve = initial_capital * (1 + daily_returns).cumprod()

# 4. 데이터프레임 구성
df = pd.DataFrame({
    'equity': equity_curve
}, index=dates)

# 인덱스 이름 설정
df.index.name = 'Date'

# 5. CSV 저장
df.to_csv('backtest_result.csv')

print("backtest_result.csv 파일이 생성되었습니다!")
print(df.head())