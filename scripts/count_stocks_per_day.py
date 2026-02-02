import pandas as pd

# CSV 파일 경로
csv_path = "data/universe/intersection_80.csv"

def count_stocks_per_day_and_monthly_avg(csv_path):
    df = pd.read_csv(csv_path)
    # 일별 종목 수
    daily_counts = df.groupby('Date').size()
    print('일별 종목 수:')
    print(daily_counts)
    # 월별 평균 종목 수
    daily_counts.index = pd.to_datetime(daily_counts.index)
    monthly_avg = daily_counts.groupby(daily_counts.index.to_period('M')).mean()
    print('\n월별 평균 종목 수:')
    print(monthly_avg)

    # 가장 많이 나온 달 5개
    print('\n가장 많이 나온 달 Top 5:')
    print(monthly_avg.sort_values(ascending=False).head(5))

    # 가장 적게 나온 달 5개
    print('\n가장 적게 나온 달 Bottom 5:')
    print(monthly_avg.sort_values().head(5))

    return daily_counts, monthly_avg

if __name__ == "__main__":
    count_stocks_per_day_and_monthly_avg(csv_path)
