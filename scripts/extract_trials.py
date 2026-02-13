import pandas as pd
import json

# ====== CONFIG ======
CONFIG = {
    "TRIALS_CSV": "results/optuna_20260210_111513/trials.csv",  # 경로 수정
    "VALUE_THRESHOLD": 400000000,  # 기준값 수정
    "OUTPUT_JSON": "best_trials.json",  # 출력 파일명 수정
}
# ====================


def main():
    csv_path = CONFIG["TRIALS_CSV"]
    value_threshold = CONFIG["VALUE_THRESHOLD"]
    output_path = CONFIG["OUTPUT_JSON"]

    df = pd.read_csv(csv_path)
    if "value" not in df.columns:
        print("trials.csv에 value 컬럼이 없습니다.")
        return

    param_cols = [c for c in df.columns if c.startswith("params_")]
    filtered = df[df["value"] > value_threshold]

    results = []
    for _, row in filtered.iterrows():
        params = {col.replace("params_", ""): row[col] for col in param_cols}
        params["value"] = row["value"]
        results.append(params)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"{len(results)} trials saved to {output_path}")


if __name__ == "__main__":
    main()
