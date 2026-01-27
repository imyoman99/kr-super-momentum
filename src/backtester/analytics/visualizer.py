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
        monthly_ret = self.df['daily_ret'].resample('M').apply(lambda x: (1 + x).prod() - 1)
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

    # 3. Monthly Return Heatmap (기간이 길어지면 높이 자동 조절)
    def plot_monthly_heatmap(self, save_path='3_monthly_heatmap.png'):
        _, m_df = self._get_monthly_df()
        pivot = m_df.pivot(index='year', columns='month', values='daily_ret')
        
        h = max(6, len(pivot) * 0.5) # 연도 수에 따라 높이 조절
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
        y_ret = self.df['daily_ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
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