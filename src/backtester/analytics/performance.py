import pandas as pd
import numpy as np

class PerformanceAnalyzer:
    def __init__(self, df: pd.DataFrame, price_col: str = 'equity'):
        """
        :param df: 백테스트 결과가 담긴 DataFrame (최소한 날짜 인덱스와 자산 가치 컬럼 필요)
        :param price_col: 자산 가치(잔고)가 기록된 컬럼명
        """
        self.df = df.copy()
        self.price_col = price_col
        
        # 일별 수익률 계산
        self.df['daily_ret'] = self.df[self.price_col].pct_change()
        
    def calculate_metrics(self, risk_free_rate: float = 0.02):
        """핵심 4가지 지표 계산"""
        
        # 1. 누적 수익률 (Cumulative Return)
        initial_val = self.df[self.price_col].iloc[0]
        final_val = self.df[self.price_col].iloc[-1]
        cumulative_ret = (final_val / initial_val) - 1
        
        # 2. CAGR (연평균 성장률)
        # 영업일 기준(252일)으로 연 단위 기간 계산
        days = (self.df.index[-1] - self.df.index[0]).days
        years = days / 365.25
        cagr = (final_val / initial_val) ** (1 / years) - 1
        
        # 3. MDD (최대 낙폭)
        rolling_max = self.df[self.price_col].cummax()
        drawdown = (self.df[self.price_col] - rolling_max) / rolling_max
        mdd = drawdown.min()
        
        # 4. Sharpe Ratio (샤프 지수)
        # 일일 무위험 수익률 환산
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_ret = self.df['daily_ret'].dropna() - daily_rf
        
        if excess_ret.std() != 0:
            sharpe_ratio = np.sqrt(252) * (excess_ret.mean() / excess_ret.std())
        else:
            sharpe_ratio = 0.0
            
        return {
            "Cumulative Return": cumulative_ret,
            "CAGR": cagr,
            "MDD": mdd,
            "Sharpe Ratio": sharpe_ratio
        }

    def print_report(self, metrics: dict):
        """결과 출력 서식"""
        print("\n" + "="*30)
        print(f"{'Backtest Performance Report':^30}")
        print("="*30)
        print(f"Total Return : {metrics['Cumulative Return']:>10.2%}")
        print(f"CAGR         : {metrics['CAGR']:>10.2%}")
        print(f"MDD          : {metrics['MDD']:>10.2%}")
        print(f"Sharpe Ratio : {metrics['Sharpe Ratio']:>10.2f}")
        print("="*30)

# --- 테스트 실행부 ---
if __name__ == "__main__":
    df = pd.read_csv('backtest_result.csv', index_col=0, parse_dates=True)
    dates = pd.date_range(start="2020-01-01", periods=500, freq='B')
    mock_equity = [1000]
    for _ in range(499):
        mock_equity.append(mock_equity[-1] * (1 + np.random.normal(0.0005, 0.01)))
    
    test_df = pd.DataFrame({'equity': mock_equity}, index=dates)
    
    # 분석 시작
    analyzer = PerformanceAnalyzer(test_df)
    results = analyzer.calculate_metrics()
    analyzer.print_report(results)