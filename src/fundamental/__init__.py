import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

DATA_DIR_PATH = r"C:\Users\dhlim\OneDrive\Desktop\kr-super-momentum\PROJECT_ROOT\data"
RS_FILE_PATH = os.path.join(DATA_DIR_PATH, "rs_list_80.csv")
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "src", "fundamental")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class FundamentalScreener:
    def __init__(self, data_folder, rs_file_path):
        self.data_folder = data_folder
        self.rs_file_path = rs_file_path
        self.price_df = self._load_price_data()
        self.rs_df = self._load_rs()

    def _load_price_data(self):
        file_list = glob.glob(os.path.join(self.data_folder, "*.parquet"))
        if not file_list:
            return pd.DataFrame()

        df_list = []
        # 컬럼 매칭을 위한 기준 리스트
        market_cols = [
            "date",
            "ticker",
            "close",
            "volume",
            "shares",
            "amount",
            "marcap",
        ]

        for file in tqdm(file_list, desc="Loading Prices"):
            try:
                temp_df = pd.read_parquet(file)
                # 1. 컬럼명 표준화 (소문자, 공백제거)
                temp_df.columns = [str(c).lower().strip() for c in temp_df.columns]

                # 2. 날짜 컬럼 확보 (가장 중요)
                # 'date'라는 컬럼이 없으면 인덱스를 리셋해서라도 만들어낸다.
                if "date" not in temp_df.columns:
                    # 인덱스 이름이 'date'나 'index'인 경우를 포함해 처리
                    temp_df = temp_df.reset_index()
                    # 리셋 후 첫 번째 컬럼(구 인덱스)을 'date'로 강제 명명
                    new_col_name = temp_df.columns[0]
                    temp_df.rename(columns={new_col_name: "date"}, inplace=True)

                # 3. 티커 컬럼 확보
                if "ticker" not in temp_df.columns:
                    # 파일명에서 티커 추출 (예: 005930.parquet -> 005930)
                    ticker_code = os.path.splitext(os.path.basename(file))[0]
                    temp_df["ticker"] = ticker_code

                # 4. 필요한 컬럼만 선택 (방어적 로직)
                # market_cols에 있거나, 재무 데이터(revenue, op_income)인 컬럼 선택
                cols_to_keep = [
                    c
                    for c in temp_df.columns
                    if c in market_cols or c in ["revenue", "op_income"]
                ]

                # [Fix] date 컬럼이 cols_to_keep에 빠져있다면 강제 추가
                if "date" not in cols_to_keep and "date" in temp_df.columns:
                    cols_to_keep.append("date")

                temp_df = temp_df[cols_to_keep]
                df_list.append(temp_df)
            except Exception:
                continue

        if not df_list:
            return pd.DataFrame()

        # 병합
        df = pd.concat(df_list, ignore_index=True)

        # [Fix] 최종 결과물에 date가 없는 경우 방어
        if "date" not in df.columns:
            return pd.DataFrame()

        # 타입 변환
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).sort_values(["ticker", "date"])

    def _load_rs(self):
        if not os.path.exists(self.rs_file_path):
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
            return rs_df
        except:
            return pd.DataFrame()

    def _get_period_label(self, date):
        m, d = date.month, date.day
        md = m * 100 + d

        # 1Q 사용: 05.16 ~ 08.15
        if 516 <= md <= 815:
            return "1Q"
        # 2Q 사용: 08.16 ~ 11.15
        elif 816 <= md <= 1115:
            return "2Q"
        # 3Q 사용: 11.16 ~ 03.31 (연말 포함)
        elif md >= 1116 or md <= 331:
            return "3Q"
        # 4Q 사용: 04.01 ~ 05.15
        elif 401 <= md <= 515:
            return "4Q"
        return "Check"

    def run(self):
        if self.price_df.empty:
            return pd.DataFrame()

        daily_df = self.price_df.copy()
        c = "close" if "close" in daily_df.columns else "Close"

        # 시총 계산
        if "marcap" not in daily_df.columns:
            daily_df["marcap"] = daily_df[c] * daily_df.get("shares", 0)

        # 거래대금 계산
        if "daily_trading_value" not in daily_df.columns:
            daily_df["daily_trading_value"] = daily_df.get(
                "amount", daily_df[c] * daily_df.get("volume", 0)
            )

        # RS 병합
        if not self.rs_df.empty:
            daily_df = pd.merge(
                daily_df,
                self.rs_df[["date", "ticker", "rs_score"]],
                on=["date", "ticker"],
                how="inner",
            )

        # 재무지표 계산 (일별 데이터에 포함된 경우)
        if "op_margin" not in daily_df.columns and "revenue" in daily_df.columns:
            daily_df["op_margin"] = np.where(
                (daily_df["revenue"] != 0) & daily_df["revenue"].notnull(),
                daily_df["op_income"] / daily_df["revenue"],
                0,
            )

        if "rev_yoy" not in daily_df.columns and "revenue" in daily_df.columns:
            daily_df["rev_yoy"] = (
                daily_df.groupby("ticker")["revenue"].pct_change(250).fillna(0)
            )

        # 기간 라벨링
        daily_df["usage_period"] = daily_df["date"].apply(self._get_period_label)

        # 필터링
        cond = (
            (daily_df["op_margin"] > 0)
            & (daily_df["rev_yoy"] > 0)
            & (daily_df["marcap"] >= 500e8)
            & (daily_df["daily_trading_value"] >= 5e8)
        )
        sel = daily_df[cond].copy()

        if sel.empty:
            return pd.DataFrame()

        # 스코어링 (Z-Score)
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
            "z_rs",
            "z_growth",
            "z_quality",
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
