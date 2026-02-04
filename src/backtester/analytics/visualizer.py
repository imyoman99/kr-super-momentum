import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, df: pd.DataFrame, price_col: str = 'equity'):
        """
        :param df: 'Date'가 인덱스이고 자산 가치 컬럼이 포함된 DataFrame
        :param price_col: 자산 가치가 기록된 컬럼명
        """
        self.df = df.copy()
        self.price_col = price_col
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)
        
        # 기초 지표 계산
        self.df['daily_ret'] = self.df[self.price_col].pct_change()
        rolling_max = self.df[self.price_col].cummax()
        self.df['drawdown'] = (self.df[self.price_col] - rolling_max) / rolling_max
        
        # 차트 스타일 설정
        plt.style.use('seaborn-v0_8-whitegrid')

    def _get_monthly_df(self):
        """월별 수익률 데이터프레임 생성 (내부용)"""
        monthly_ret = self.df['daily_ret'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
        m_df = monthly_ret.to_frame()
        m_df['year'] = m_df.index.year
        m_df['month'] = m_df.index.month
        return monthly_ret, m_df

    # 1. Equity Curve
    def plot_equity_curve(self, save_path='1_equity_curve.png', log_scale=False):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df[self.price_col], color='#1f77b4', lw=2)
        if log_scale:
            plt.yscale('log')
            plt.title('Equity Curve (Log Scale)', fontsize=14, fontweight='bold')
        else:
            plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # 2. Drawdown Curve
    def plot_drawdown(self, save_path='2_drawdown.png'):
        plt.figure(figsize=(12, 5))
        plt.fill_between(self.df.index, self.df['drawdown'] * 100, 0, color='#d62728', alpha=0.3)
        plt.plot(self.df.index, self.df['drawdown'] * 100, color='#d62728', lw=1)
        plt.title('Drawdown (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Percentage (%)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # 3. Monthly Return Heatmap 
    def plot_monthly_heatmap(self, save_path='3_monthly_heatmap.png'):
        _, m_df = self._get_monthly_df()
        pivot = m_df.pivot(index='year', columns='month', values='daily_ret')
        
        h = max(6, len(pivot) * 0.5) 
        plt.figure(figsize=(12, h))
        sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap='RdYlGn', center=0, cbar_kws={'label': 'Return (%)'})
        plt.title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # 4. Average Monthly Return (Seasonality)
    def plot_seasonality(self, save_path='4_seasonality.png'):
        m_ret, _ = self._get_monthly_df()
        avg_m = m_ret.groupby(m_ret.index.month).mean() * 100
        avg_m.index = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(avg_m)]
        
        plt.figure(figsize=(10, 6))
        colors = ['#d62728' if x < 0 else '#2ca02c' for x in avg_m]
        avg_m.plot(kind='bar', color=colors, alpha=0.8)
        plt.title('Average Monthly Return (%)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # 5. Yearly Return
    def plot_yearly_return(self, save_path='5_yearly_return.png'):
        y_ret = self.df['daily_ret'].resample('YE').apply(lambda x: (1 + x).prod() - 1) * 100
        y_ret.index = y_ret.index.year
        
        plt.figure(figsize=(10, 6))
        colors = ['#d62728' if x < 0 else '#2ca02c' for x in y_ret]
        y_ret.plot(kind='bar', color=colors, alpha=0.8)
        plt.title('Yearly Return (%)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # 6. Yearly Monthly Return (Grouped Bar)
    def plot_yearly_monthly_bar(self, save_path='6_yearly_monthly_bar.png'):
        _, m_df = self._get_monthly_df()
        plt.figure(figsize=(14, 7))
        sns.barplot(data=m_df, x='month', y='daily_ret', hue='year', palette='tab10')
        plt.title('Monthly Returns Comparison by Year', fontsize=14, fontweight='bold')
        plt.axhline(0, color='black', lw=1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Year')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # 7. Daily Return Distribution
    def plot_return_dist(self, save_path='7_return_dist.png'):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['daily_ret'].dropna() * 100, kde=True, color='#9467bd')
        plt.title('Daily Return Distribution (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Return (%)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# visualizer.py 내 Visualizer 클래스 안에 추가

    def plot_trade_analysis(self, ticker, entry_date, df_ticker, save_path):
        """
        개별 종목의 진입 시점과 기술적 지표(이평선, 거래량 수축)를 시각화합니다.
        :param ticker: 종목명 또는 코드
        :param entry_date: 매수일 (pd.Timestamp)
        :param df_ticker: 해당 종목의 전체 시세 데이터프레임 (MA, Volume 포함)
        :param save_path: 저장 경로
        """
        # 진입일 기준 전후 60일 정도의 데이터만 보기 좋게 슬라이싱
        start_date = entry_date - pd.Timedelta(days=60)
        end_date = entry_date + pd.Timedelta(days=30)
        df = df_ticker.loc[start_date:end_date].copy()

        if df.empty:
            return

        # 차트 생성 (가격/이평선 섹션 + 거래량 섹션)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})

        # --- 상단: 가격 및 이동평균선 ---
        ax1.plot(df.index, df['Close'], color='black', lw=1.5, label='Close')
        if 'MA50' in df.columns: ax1.plot(df.index, df['MA50'], label='MA50', alpha=0.8)
        if 'MA150' in df.columns: ax1.plot(df.index, df['MA150'], label='MA150', alpha=0.8)
        if 'MA200' in df.columns: ax1.plot(df.index, df['MA200'], label='MA200', linestyle='--', alpha=0.8)
        
        # 매수 지점 표시 (화살표)
        if entry_date in df.index:
            ax1.annotate('BUY ENTRY', xy=(entry_date, df.loc[entry_date, 'Close']),
                         xytext=(entry_date, df.loc[entry_date, 'Close'] * 0.85),
                         arrowprops=dict(facecolor='green', shrink=0.05, width=2),
                         fontsize=10, fontweight='bold', color='green', ha='center')
        
        ax1.set_title(f"Trade Analysis: {ticker} (Entry: {entry_date.date()})", fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')

        # --- 하단: 거래량 및 수축(Contraction) 확인 ---
        colors = ['red' if df.iloc[i]['Close'] > df.iloc[i-1]['Close'] else 'blue' 
                  for i in range(1, len(df))]
        colors.insert(0, 'gray') # 첫날 색상
        
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.4)
        
        # 거래량 20일 이평선 (수축 판단 기준선)
        vol_ma = df['Volume'].rolling(window=20).mean()
        ax2.plot(df.index, vol_ma, color='purple', lw=1, label='Vol MA20', linestyle=':')
        
        # [작성자 요청 반영] 거래량 수축 구간 강조 (최근 10일 평균 < 이전 10일 평균 조건 시각화)
        # 여기서는 단순히 평균 거래량보다 낮은 날들을 시각적으로 강조
        ax2.fill_between(df.index, 0, df['Volume'].max(), 
                         where=(df['Volume'] < vol_ma * 0.7), 
                         color='yellow', alpha=0.2, label='Vol Contraction Area')

        ax2.set_ylabel("Volume")
        ax2.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
# --- 테스트 실행부 ---
if __name__ == "__main__":
    # 1. 데이터 로드
    try:
        df = pd.read_csv('backtest_result.csv', index_col=0, parse_dates=True)
        
        # 2. 시각화 객체 생성
        vis = Visualizer(df, price_col='equity')
        
        # 3. 주요 차트 생성 테스트
        print("차트 생성 중...")
        vis.plot_equity_curve('test_equity.png')        # 자산 곡선
        vis.plot_drawdown('test_drawdown.png')          # 낙폭
        vis.plot_monthly_heatmap('test_heatmap.png')    # 월별 히트맵
        vis.plot_seasonality('test_seasonality.png')    # 계절성 평균
        vis.plot_yearly_return('test_yearly.png')       # 연도별 수익률
        vis.plot_yearly_monthly_bar('test_yearly_monthly.png') # 연도별 월간 비교
        vis.plot_return_dist('test_return_dist.png')    # 일별 수익률 분포
        print("모든 테스트 차트가 현재 폴더에 저장되었습니다.")
        
    except FileNotFoundError:
        print("Error: 'backtest_result.csv' 파일이 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류 발생: {e}")