import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

DATA_DIR_PATH = r"C:\Users\dhlim\OneDrive\Desktop\kr-super-momentum\PROJECT_ROOT\data"
RS_FILE_PATH = os.path.join(DATA_DIR_PATH, "intersection_rs.csv")
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "src", "fundamental")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class FundamentalScreener:
    def __init__(self, data_folder, rs_file_path):
        self.data_folder = data_folder
        self.rs_file_path = rs_file_path
        print(">>> [Init] Loading Data...")
        self.price_df = self._load_price_data()
        self.rs_df = self._load_rs()

    def _load_price_data(self):
        file_list = glob.glob(os.path.join(self.data_folder, "*.parquet"))
        if not file_list:
            print(f"!!! Error: No parquet files found in {self.data_folder}")
            return pd.DataFrame()

        print(f"Found {len(file_list)} parquet files.")

        df_list = []
        market_cols = [
            "date",
            "ticker",
            "close",
            "volume",
            "shares",
            "amount",
            "marcap",
            "name",
        ]

        for file in tqdm(file_list, desc="Loading Prices"):
            try:
                temp_df = pd.read_parquet(file)
                temp_df.columns = [str(c).lower().strip() for c in temp_df.columns]

                if "date" not in temp_df.columns:
                    temp_df = temp_df.reset_index()
                    temp_df.rename(columns={temp_df.columns[0]: "date"}, inplace=True)

                if "ticker" not in temp_df.columns:
                    if "code" in temp_df.columns:
                        temp_df.rename(columns={"code": "ticker"}, inplace=True)
                    else:
                        ticker_code = os.path.splitext(os.path.basename(file))[0]
                        temp_df["ticker"] = ticker_code

                temp_df["ticker"] = temp_df["ticker"].astype(str).str.zfill(6)

                cols_to_keep = [
                    c
                    for c in temp_df.columns
                    if c in market_cols or c in ["revenue", "op_income"]
                ]
                if "date" not in cols_to_keep and "date" in temp_df.columns:
                    cols_to_keep.append("date")

                temp_df = temp_df[cols_to_keep]
                df_list.append(temp_df)
            except Exception as e:
                # print(f"Error reading {file}: {e}")
                continue

        if not df_list:
            return pd.DataFrame()

        df = pd.concat(df_list, ignore_index=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values(["ticker", "date"])

        print(
            f">>> [Price Data] Loaded {len(df)} rows. Sample Ticker: {df['ticker'].iloc[0]}"
        )
        return df

    def _load_rs(self):
        if not os.path.exists(self.rs_file_path):
            print(f"!!! Error: RS file not found at {self.rs_file_path}")
            return pd.DataFrame()
        try:
            rs_df = pd.read_csv(self.rs_file_path)
            rs_df.rename(
                columns={"Date": "date", "Code": "ticker", "RS": "rs_score"},
                inplace=True,
            )
            rs_df["date"] = pd.to_datetime(rs_df["date"])

            rs_df["ticker"] = rs_df["ticker"].apply(
                lambda x: f"{int(x):06d}" if str(x).isdigit() else str(x)
            )

            print(
                f">>> [RS Data] Loaded {len(rs_df)} rows. Sample Ticker: {rs_df['ticker'].iloc[0]}"
            )
            return rs_df
        except Exception as e:
            print(f"!!! Error loading RS file: {e}")
            return pd.DataFrame()

    def _get_period_label(self, date):
        m, d = date.month, date.day
        md = m * 100 + d
        if 516 <= md <= 815:
            return "1Q"
        elif 816 <= md <= 1115:
            return "2Q"
        elif md >= 1116 or md <= 331:
            return "3Q"
        elif 401 <= md <= 515:
            return "4Q"
        return "Check"

    def run(self):
        if self.price_df.empty:
            print("!!! Price DF is empty.")
            return pd.DataFrame()

        daily_df = self.price_df.copy()
        print(f"Step 1: Start with {len(daily_df)} rows from Price Data")

        c = "close" if "close" in daily_df.columns else "Close"
        if "marcap" not in daily_df.columns:
            daily_df["marcap"] = daily_df[c] * daily_df.get("shares", 0)

        if "daily_trading_value" not in daily_df.columns:
            daily_df["daily_trading_value"] = daily_df.get(
                "amount", daily_df[c] * daily_df.get("volume", 0)
            )

        if not self.rs_df.empty:
            before_merge = len(daily_df)
            daily_df = pd.merge(
                daily_df,
                self.rs_df[["date", "ticker", "rs_score"]],
                on=["date", "ticker"],
                how="inner",
            )
            print(
                f"Step 2: After RS Merge (Inner Join) -> {len(daily_df)} rows (Dropped {before_merge - len(daily_df)} rows)"
            )

            if daily_df.empty:
                print("!!! Merge failed.")
                return pd.DataFrame()

        if "revenue" in daily_df.columns:
            if "op_margin" not in daily_df.columns:
                daily_df["op_margin"] = np.where(
                    (daily_df["revenue"] != 0) & daily_df["revenue"].notnull(),
                    daily_df["op_income"] / daily_df["revenue"],
                    0,
                )
            if "rev_yoy" not in daily_df.columns:
                daily_df["rev_yoy"] = (
                    daily_df.groupby("ticker")["revenue"].pct_change(250).fillna(0)
                )
        else:
            print("!!! Warning: 'revenue' column missing. Financial filters will fail.")
            daily_df["op_margin"] = 0
            daily_df["rev_yoy"] = 0

        daily_df["usage_period"] = daily_df["date"].apply(self._get_period_label)

        # 연도 추출
        daily_df["year"] = daily_df["date"].dt.year

        print(f"Step 3: Filtering...")

        # 1. 영업이익률 > 0 (공통 조건)
        cond_quality = daily_df["op_margin"] > 0

        # 2. 성장성 조건 (16~20년은 제외, 21년부터는 rev_yoy > 0)
        #    조건: (연도가 2016~2020 사이임) OR (rev_yoy > 0)
        cond_growth = (daily_df["year"].between(2016, 2020)) | (daily_df["rev_yoy"] > 0)

        # 3. 유동성 조건
        cond_liquidity = (daily_df["marcap"] >= 500e8) & (
            daily_df["daily_trading_value"] >= 5e8
        )

        # 최종 필터링
        sel = daily_df[cond_quality & cond_growth & cond_liquidity].copy()
        print(f" - Final Filtered Rows: {len(sel)}")

        if sel.empty:
            print("!!! Result is empty after filtering.")
            return pd.DataFrame()

        def get_zscore(x):
            if x.std() == 0:
                return 0
            return (x - x.mean()) / x.std()

        g = sel.groupby("date")
        sel["z_rs"] = g["rs_score"].transform(get_zscore)
        sel["z_growth"] = g["rev_yoy"].transform(get_zscore)
        sel["z_quality"] = g["op_margin"].transform(get_zscore)

        for col in ["z_rs", "z_growth", "z_quality"]:
            sel[col] = sel[col].clip(-3, 3)

        sel["raw_total_score"] = (
            (sel["z_rs"] * 0.5) + (sel["z_growth"] * 0.3) + (sel["z_quality"] * 0.2)
        )

        g_final = sel.groupby("date")["raw_total_score"]
        min_s = g_final.transform("min")
        max_s = g_final.transform("max")
        sel["total_score"] = (
            (sel["raw_total_score"] - min_s) / (max_s - min_s + 1e-9) * 100
        )

        out_cols = [
            "date",
            "usage_period",
            "ticker",
            "name",
            "total_score",
            "rs_score",
            "rev_yoy",
            "op_margin",
            "marcap",
            "daily_trading_value",
        ]
        final_cols = [c for c in out_cols if c in sel.columns]

        return sel[final_cols].sort_values(
            by=["date", "total_score"], ascending=[True, False]
        )


if __name__ == "__main__":
    app = FundamentalScreener(DATA_DIR_PATH, RS_FILE_PATH)
    res = app.run()

    if not res.empty:
        path = os.path.join(OUTPUT_DIR, "scored_strategy_result_daily.csv")
        res.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved: {path} ({len(res)} rows)")
    else:
        print("No results found.")
