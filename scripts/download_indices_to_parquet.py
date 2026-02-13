import FinanceDataReader as fdr
import pandas as pd

# 저장할 대상: (설명, 티커, 파일명)
targets = [
    ("코스피 종합", "KS11"),
    ("코스닥 종합", "KQ11"),
    ("코스피 중형주 (TIGER)", "277650"),
    ("코스피 중소형 (KODEX)", "226980"),
    ("코스피200 동일가중 (KODEX)", "252650"),
]

COLUMNS = [
    "Code",
    "Name",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Change",
    "UpDown",
    "Comp",
    "Amount",
    "MarCap",
    "RS",
    "Date",
]
START = "2016-01-01"
END = "2025-12-31"

for desc, ticker in targets:
    print(f"Downloading {desc} ({ticker}) ...")
    df = fdr.DataReader(ticker, START, END)
    if df.empty:
        print(f"  [!] No data for {ticker}")
        continue

    df = df.reset_index()
    df["Code"] = ticker
    df["Name"] = desc

    # 이동평균선 추가
    ma_windows = [5, 20, 50, 60, 120, 150, 200, 240]
    for win in ma_windows:
        df[f"MA{win}"] = df["Close"].rolling(window=win).mean()

    # 칼럼이 없으면 None으로 채움
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    # MA 칼럼 추가
    for win in ma_windows:
        col = f"MA{win}"
        if col not in df.columns:
            df[col] = None
    # 칼럼 순서 맞추기: MA는 MarCap 다음, Date는 마지막
    ma_cols = [f"MA{win}" for win in ma_windows]
    # COLUMNS: ... 'MarCap', 'RS', 'Date', ...
    base_cols = COLUMNS.copy()
    marcap_idx = base_cols.index("MarCap")
    rs_idx = base_cols.index("RS")
    # MarCap 다음에 MA, 그 다음 RS, 마지막에 Date
    out_cols = (
        base_cols[: marcap_idx + 1]
        + ma_cols
        + base_cols[rs_idx : rs_idx + 1]
        + [col for col in base_cols[rs_idx + 1 :] if col != "Date"]
        + ["Date"]
    )
    df = df[out_cols]
    fname = f"{ticker}.parquet"
    df.to_parquet(fname, index=False)
    print(f"  -> Saved to {fname}")
