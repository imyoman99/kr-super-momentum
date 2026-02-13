import argparse
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트 경로를 sys.path에 추가
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtester.strategies.minervini_filter import (
    check_minervini_from_df,
    load_from_parquet,
)


def normalize_code(code_value: object) -> str:
    return f"{int(code_value):06d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a universe CSV using Minervini rules."
    )
    parser.add_argument(
        "--input",
        default="data/universe/intersection_80.csv",
        help="Input universe CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/universe/intersection_80_minervini4.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing <code>.parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    universe = pd.read_csv(input_path)
    if "Date" not in universe.columns or "Code" not in universe.columns:
        raise ValueError("Input CSV must contain Date and Code columns")

    universe["Date"] = pd.to_datetime(universe["Date"])

    cache: dict[str, pd.DataFrame | None] = {}
    keep_mask: list[bool] = []

    for row in universe.itertuples(index=False):
        code = normalize_code(row.Code)
        df = cache.get(code)
        if df is None and code not in cache:
            df = load_from_parquet(code, data_dir=args.data_dir)
            cache[code] = df

        keep_mask.append(check_minervini_from_df(df, row.Date))

    filtered = universe[keep_mask].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)

    print(f"Input rows: {len(universe)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
