import pandas as pd
import numpy as np

def get_cagr(equity_curve):
    total_return = equity_curve[-1] / equity_curve[0]
    years = len(equity_curve) / 252 # 영업일 기준
    return (total_return ** (1 / years)) - 1

def get_mdd(equity_curve):
    drawdown = equity_curve / equity_curve.cummax() - 1
    return drawdown.min()

def get_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()