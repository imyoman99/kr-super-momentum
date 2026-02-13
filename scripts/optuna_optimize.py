import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "optuna is not installed. Install with: pip install optuna"
    ) from exc

import importlib.util

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "backtester"
    / "strategies"
    / "Asset_Preprocessing.py"
)


spec = importlib.util.spec_from_file_location("ap", MODULE_PATH)
if spec is None:
    raise ImportError(f"[IMPORT ERROR] MODULE_PATH not found: {MODULE_PATH}")
ap = importlib.util.module_from_spec(spec)
if spec.loader is None:
    raise ImportError(f"[IMPORT ERROR] spec.loader is None for: {MODULE_PATH}")
spec.loader.exec_module(ap)

BASE_CONFIG = copy.deepcopy(ap.CONFIG)

OPTUNA_CONFIG = {
    "BEST_CONFIG_FILENAME": "best_config.json",
    "TRIALS_FILENAME": "trials.csv",
    "FINAL_MDD_PCT": -30.0,
    "RUN_DIR": r"C:\Users\dudals\Downloads\Team_Project\kr_super_momentum\results\optuna_20260211_013037",
    "N_TRIALS": 200,
    "QUICK_TRAIN": ("2016-01-01", "2016-12-31"),
    "QUICK_MDD_CUT": -60.0,
    "MIN_TRAIN_TRADES": 12,
    "TRAIN_MDD_CUT": -99.0,
    "WALK_FORWARD": {
        "TRAIN": ("2016-01-01", "2020-12-31"),
        "VALID": ("2021-01-01", "2022-12-31"),
        "TEST": ("2023-01-01", "2025-12-31"),
    },
    "FIXED_PARAMS": {
        "STOP_ATR_MULTIPLE": 2.0,
        "RISK_PER_TRADE": 0.01,
        "TOTAL_RISK_CAP": 0.15,
        "COST_RATE": BASE_CONFIG.get("COST_RATE"),
        "SIGNAL_WINDOW_MODE": "next_day",
    },
    "REGIME_OPTIONS": {
        "USE_REGIME_FILTER": [False],
        "REGIME_SELECTED_TICKERS": (
            BASE_CONFIG.get("REGIME_TICKERS")
            or BASE_CONFIG.get("REGIME_SELECTED_TICKERS")
            or []
        ),
        "REGIME_SELECTED_REQUIREMENT": ["any"],
    },
    "PARAM_RANGES": {
        "TAKE_PROFIT_R": (1.5, 4.0, 0.5),
        "CHANDELIER_ATR_MULTIPLE": (2.5, 4.5, 0.5),
        "ENTRY_VALUE_LOOKBACK": (10, 30, 5),
        "ENTRY_VALUE_MULTIPLE": (0.9, 1.2, 0.05),
        "MAX_POSITION_RATIO": (0.1, 0.3, 0.05),
        "ATR_FLOOR_RATIO": (0.005, 0.015, 0.005),
    },
}

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_CONFIG_FILENAME = OPTUNA_CONFIG["BEST_CONFIG_FILENAME"]
TRIALS_FILENAME = OPTUNA_CONFIG["TRIALS_FILENAME"]
EXISTING_PARAM_HASHES = set()
RESULT_CACHE = {}


def _resolve_run_dir(run_dir_override=None):
    if run_dir_override:
        run_dir = Path(run_dir_override)
        run_dir.mkdir(parents=True, exist_ok=True)
        trials_path = run_dir / TRIALS_FILENAME
        return run_dir, trials_path if trials_path.exists() else None
    new_dir = RESULTS_DIR / f"optuna_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir, None


def _compute_param_hash(params, precision=3):
    normalized = {}
    for key in sorted(params.keys()):
        val = params[key]
        if isinstance(val, float):
            normalized[key] = f"{val:.{precision}f}"
        else:
            normalized[key] = str(val)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _row_params_from_df(df_row, param_cols):
    params = {}
    for col in param_cols:
        key = col.replace("params_", "")
        val = df_row.get(col)
        if val is None:
            continue
        params[key] = val
    if "ENTRY_VALUE_LOOKBACK" in params:
        params["ENTRY_VALUE_LOOKBACK"] = int(params["ENTRY_VALUE_LOOKBACK"])
    return params


def _ensure_hash_column(csv_path, precision=3):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "param_hash" in df.columns:
        return

    param_cols = [c for c in df.columns if c.startswith("params_")]
    df["param_hash"] = df.apply(
        lambda row: _compute_param_hash(
            _row_params_from_df(row, param_cols),
            precision=precision,
        ),
        axis=1,
    )
    df.to_csv(csv_path, index=False)


def _get_best_value_from_csv(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "value" not in df.columns or df.empty:
        return None
    return float(df["value"].max())


def _load_trials_from_csv(study, csv_path):
    import pandas as pd
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )

    all_distributions = _build_distributions(
        CategoricalDistribution, FloatDistribution, IntDistribution
    )

    df = pd.read_csv(csv_path)
    if "value" not in df.columns:
        return 0

    param_cols = [c for c in df.columns if c.startswith("params_")]
    added = 0
    hashes = set()
    for _, row in df.iterrows():
        if pd.isna(row.get("value")):
            continue
        params = _row_params_from_df(row, param_cols)
        distributions = {
            key: all_distributions[key]
            for key in params.keys()
            if key in all_distributions
        }
        if set(params.keys()) != set(distributions.keys()):
            continue

        try:
            for key in ["ENTRY_VALUE_LOOKBACK"]:
                if key in params:
                    params[key] = int(params[key])

            trial = optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=float(row["value"]),
            )
            study.add_trial(trial)
            added += 1
            hashes.add(_compute_param_hash(params))
        except Exception:
            continue
    return added, hashes


def _append_trials_csv(trials_path, new_trials_df, precision=3):
    if trials_path.exists():
        _ensure_hash_column(trials_path, precision=precision)
        next_num = _get_next_trial_number(trials_path)
        if "number" in new_trials_df.columns:
            new_trials_df = new_trials_df.copy()
            new_trials_df["number"] = range(next_num, next_num + len(new_trials_df))
        new_trials_df.to_csv(trials_path, mode="a", index=False, header=False)
    else:
        new_trials_df.to_csv(trials_path, index=False)


def _get_next_trial_number(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return 0
    if "number" in df.columns and df["number"].notna().any():
        return int(df["number"].max()) + 1
    return int(len(df))


def _range_parts(range_value):
    if isinstance(range_value, (list, tuple)) and len(range_value) == 3:
        return range_value[0], range_value[1], range_value[2]
    if isinstance(range_value, (list, tuple)) and len(range_value) == 2:
        return range_value[0], range_value[1], None
    raise ValueError(f"Invalid range definition: {range_value}")


def _build_distributions(CategoricalDistribution, FloatDistribution, IntDistribution):
    ranges = OPTUNA_CONFIG["PARAM_RANGES"]
    regime_opts = OPTUNA_CONFIG["REGIME_OPTIONS"]
    tp_min, tp_max, tp_step = _range_parts(ranges["TAKE_PROFIT_R"])
    ch_min, ch_max, ch_step = _range_parts(ranges["CHANDELIER_ATR_MULTIPLE"])
    mpr_min, mpr_max, mpr_step = _range_parts(ranges["MAX_POSITION_RATIO"])
    evl_min, evl_max, evl_step = _range_parts(ranges["ENTRY_VALUE_LOOKBACK"])
    evm_min, evm_max, evm_step = _range_parts(ranges["ENTRY_VALUE_MULTIPLE"])
    atr_min, atr_max, atr_step = _range_parts(ranges["ATR_FLOOR_RATIO"])
    return {
        "TAKE_PROFIT_R": FloatDistribution(tp_min, tp_max, step=tp_step),
        "CHANDELIER_ATR_MULTIPLE": FloatDistribution(ch_min, ch_max, step=ch_step),
        "MAX_POSITION_RATIO": FloatDistribution(mpr_min, mpr_max, step=mpr_step),
        "ENTRY_VALUE_LOOKBACK": IntDistribution(evl_min, evl_max, step=evl_step),
        "ENTRY_VALUE_MULTIPLE": FloatDistribution(evm_min, evm_max, step=evm_step),
        "ATR_FLOOR_RATIO": FloatDistribution(atr_min, atr_max, step=atr_step),
        "USE_REGIME_FILTER": CategoricalDistribution(regime_opts["USE_REGIME_FILTER"]),
        "REGIME_SELECTED_TICKER": CategoricalDistribution(
            regime_opts["REGIME_SELECTED_TICKERS"]
        ),
        "REGIME_SELECTED_REQUIREMENT": CategoricalDistribution(
            regime_opts["REGIME_SELECTED_REQUIREMENT"]
        ),
    }


def apply_trial_config(trial):
    ranges = OPTUNA_CONFIG["PARAM_RANGES"]
    regime_opts = OPTUNA_CONFIG["REGIME_OPTIONS"]
    tp_min, tp_max, tp_step = _range_parts(ranges["TAKE_PROFIT_R"])
    ch_min, ch_max, ch_step = _range_parts(ranges["CHANDELIER_ATR_MULTIPLE"])
    mpr_min, mpr_max, mpr_step = _range_parts(ranges["MAX_POSITION_RATIO"])
    evl_min, evl_max, evl_step = _range_parts(ranges["ENTRY_VALUE_LOOKBACK"])
    evm_min, evm_max, evm_step = _range_parts(ranges["ENTRY_VALUE_MULTIPLE"])
    atr_min, atr_max, atr_step = _range_parts(ranges["ATR_FLOOR_RATIO"])
    fixed_params = OPTUNA_CONFIG.get("FIXED_PARAMS", {})
    ap.CONFIG["STOP_ATR_MULTIPLE"] = fixed_params.get(
        "STOP_ATR_MULTIPLE", ap.CONFIG.get("STOP_ATR_MULTIPLE")
    )
    ap.CONFIG["RISK_PER_TRADE"] = fixed_params.get(
        "RISK_PER_TRADE", ap.CONFIG.get("RISK_PER_TRADE")
    )
    ap.CONFIG["TOTAL_RISK_CAP"] = fixed_params.get(
        "TOTAL_RISK_CAP", ap.CONFIG.get("TOTAL_RISK_CAP")
    )
    ap.CONFIG["COST_RATE"] = fixed_params.get("COST_RATE", ap.CONFIG.get("COST_RATE"))
    ap.CONFIG["SIGNAL_WINDOW_MODE"] = fixed_params.get(
        "SIGNAL_WINDOW_MODE", ap.CONFIG.get("SIGNAL_WINDOW_MODE")
    )
    ap.CONFIG["TAKE_PROFIT_R"] = trial.suggest_float(
        "TAKE_PROFIT_R", tp_min, tp_max, step=tp_step
    )
    ap.CONFIG["CHANDELIER_ATR_MULTIPLE"] = trial.suggest_float(
        "CHANDELIER_ATR_MULTIPLE", ch_min, ch_max, step=ch_step
    )
    ap.CONFIG["ATR_FLOOR_RATIO"] = trial.suggest_float(
        "ATR_FLOOR_RATIO", atr_min, atr_max, step=atr_step
    )
    ap.CONFIG["MAX_POSITION_RATIO"] = trial.suggest_float(
        "MAX_POSITION_RATIO", mpr_min, mpr_max, step=mpr_step
    )
    ap.CONFIG["ENTRY_VALUE_LOOKBACK"] = trial.suggest_int(
        "ENTRY_VALUE_LOOKBACK", evl_min, evl_max, step=evl_step
    )
    ap.CONFIG["ENTRY_VALUE_MULTIPLE"] = trial.suggest_float(
        "ENTRY_VALUE_MULTIPLE", evm_min, evm_max, step=evm_step
    )

    ap.CONFIG["USE_REGIME_FILTER"] = trial.suggest_categorical(
        "USE_REGIME_FILTER", regime_opts["USE_REGIME_FILTER"]
    )
    if ap.CONFIG["USE_REGIME_FILTER"]:
        if regime_opts["REGIME_SELECTED_TICKERS"]:
            selected_ticker = trial.suggest_categorical(
                "REGIME_SELECTED_TICKER", regime_opts["REGIME_SELECTED_TICKERS"]
            )
            ap.CONFIG["REGIME_SELECTED_TICKERS"] = [selected_ticker]
        ap.CONFIG["REGIME_SELECTED_REQUIREMENT"] = trial.suggest_categorical(
            "REGIME_SELECTED_REQUIREMENT",
            regime_opts["REGIME_SELECTED_REQUIREMENT"],
        )
    else:
        ap.CONFIG["REGIME_SELECTED_TICKERS"] = []
        ap.CONFIG["REGIME_SELECTED_REQUIREMENT"] = None


def objective(trial):
    ap.CONFIG.clear()
    ap.CONFIG.update(copy.deepcopy(BASE_CONFIG))
    apply_trial_config(trial)

    use_regime = ap.CONFIG.get("USE_REGIME_FILTER")
    regime_tickers = ap.CONFIG.get("REGIME_SELECTED_TICKERS") if use_regime else None
    regime_requirement = ap.CONFIG.get("REGIME_SELECTED_REQUIREMENT")

    param_hash = _compute_param_hash(trial.params)
    if param_hash in EXISTING_PARAM_HASHES:
        raise optuna.TrialPruned("duplicate-params")
    EXISTING_PARAM_HASHES.add(param_hash)

    wf = OPTUNA_CONFIG.get("WALK_FORWARD", {})
    quick_start, quick_end = OPTUNA_CONFIG["QUICK_TRAIN"]

    # --- 1. QUICK TEST (2016) ---
    ap.CONFIG["START_DATE"], ap.CONFIG["END_DATE"] = quick_start, quick_end
    quick_key = f"{param_hash}_quick"
    if quick_key in RESULT_CACHE:
        quick_metrics = RESULT_CACHE[quick_key]
    else:
        try:
            _, _, quick_metrics = ap.run_backtest(
                save_outputs=False,
                silent=True,
                use_regime_filter=use_regime,
                regime_tickers=regime_tickers,
                regime_requirement=regime_requirement,
            )
            RESULT_CACHE[quick_key] = quick_metrics
        except Exception as exc:
            print(f"[ERROR] Backtest Failed: {exc}")
            return -9999

    q_mdd = float(quick_metrics.get("MDD", 0.0))
    q_trades = int(quick_metrics.get("Trades", 0))

    if q_mdd < OPTUNA_CONFIG["QUICK_MDD_CUT"]:
        print(
            f"[PRUNED] Quick MDD Fail: {q_mdd:.4f} < {OPTUNA_CONFIG['QUICK_MDD_CUT']}"
        )
        raise optuna.TrialPruned(f"early_fail_mdd_{q_mdd:.2f}")

    # --- 2. TRAIN TEST (2016-2020) ---
    ap.CONFIG["START_DATE"], ap.CONFIG["END_DATE"] = wf.get(
        "TRAIN", (ap.CONFIG.get("START_DATE"), ap.CONFIG.get("END_DATE"))
    )
    train_key = f"{param_hash}_train"
    if train_key in RESULT_CACHE:
        train_metrics = RESULT_CACHE[train_key]
    else:
        _, _, train_metrics = ap.run_backtest(
            save_outputs=False,
            silent=True,
            use_regime_filter=use_regime,
            regime_tickers=regime_tickers,
            regime_requirement=regime_requirement,
        )
        RESULT_CACHE[train_key] = train_metrics

    t_trades = int(train_metrics.get("Trades", 0))

    if t_trades < OPTUNA_CONFIG["MIN_TRAIN_TRADES"]:
        print(
            f"[PRUNED] Too Few Trades: {t_trades} < {OPTUNA_CONFIG['MIN_TRAIN_TRADES']}"
        )
        raise optuna.TrialPruned(f"too_few_trades_{t_trades}")

    # --- 3. VALID TEST (2021-2022) ---
    ap.CONFIG["START_DATE"], ap.CONFIG["END_DATE"] = wf.get(
        "VALID", (ap.CONFIG.get("START_DATE"), ap.CONFIG.get("END_DATE"))
    )
    valid_key = f"{param_hash}_valid"
    if valid_key in RESULT_CACHE:
        metrics = RESULT_CACHE[valid_key]
    else:
        _, _, metrics = ap.run_backtest(
            save_outputs=False,
            silent=True,
            use_regime_filter=use_regime,
            regime_tickers=regime_tickers,
            regime_requirement=regime_requirement,
        )
        RESULT_CACHE[valid_key] = metrics

    cagr = float(metrics.get("CAGR", 0.0))
    mdd = abs(float(metrics.get("MDD", 0.0)))

    if mdd <= 0:
        print(f"[PRUNED] Invalid MDD (Zero): {mdd}")
        raise optuna.TrialPruned("invalid_mdd")

    effective_mdd = max(mdd, 0.15)
    score = cagr / effective_mdd
    print(
        f"[OK] Trial Finished. Score: {score:.4f} "
        f"(CAGR: {cagr:.2f}, MDD: {mdd:.2f}, Eff.MDD: {effective_mdd:.2f})"
    )
    return score


def main():
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=30,
        multivariate=True,
        group=True,
        constant_liar=True,
        seed=42,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=50,
        n_warmup_steps=0,
        interval_steps=1,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    run_dir, latest_trials = _resolve_run_dir(OPTUNA_CONFIG.get("RUN_DIR"))
    prior_trials = 0
    prior_hashes = set()
    existing_best_value = None
    if latest_trials:
        _ensure_hash_column(latest_trials)
        # _load_trials_from_csv가 항상 (int, set) 반환하도록 안전하게 처리
        result = _load_trials_from_csv(study, latest_trials)
        if isinstance(result, tuple) and len(result) == 2:
            prior_trials, prior_hashes = result
        else:
            prior_trials, prior_hashes = 0, set()
        existing_best_value = _get_best_value_from_csv(latest_trials)
        if prior_trials:
            EXISTING_PARAM_HASHES.update(prior_hashes)

    if prior_trials:
        print(f"[OPTUNA] Loaded prior trials: {prior_trials}")
        print(f"[OPTUNA] Best before optimize: {study.best_value}")
    elif latest_trials:
        print("[OPTUNA] No valid trials found in latest trials.csv")
    before_count = len(study.trials)
    study.optimize(objective, n_trials=OPTUNA_CONFIG["N_TRIALS"])

    trials_path = run_dir / TRIALS_FILENAME
    new_trials_df = study.trials_dataframe().iloc[before_count:].copy()
    if not new_trials_df.empty:
        param_cols = [c for c in new_trials_df.columns if c.startswith("params_")]
        new_trials_df["param_hash"] = new_trials_df.apply(
            lambda row: _compute_param_hash(_row_params_from_df(row, param_cols)),
            axis=1,
        )
    _append_trials_csv(trials_path, new_trials_df)

    try:
        best_value = study.best_value
        best_params = study.best_params
    except ValueError:
        print("[OPTUNA] No completed trials. Skipping best config update.")
        best_value = None
        best_params = None

    should_update_best = best_value is not None and (
        existing_best_value is None or best_value > existing_best_value
    )
    if should_update_best and best_params is not None:
        best_config = copy.deepcopy(BASE_CONFIG)
        best_config.update(best_params)
        if not best_config.get("USE_REGIME_FILTER"):
            best_config["REGIME_SELECTED_TICKERS"] = []
            best_config["REGIME_SELECTED_REQUIREMENT"] = None
        (run_dir / BEST_CONFIG_FILENAME).write_text(
            json.dumps(best_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        wf = OPTUNA_CONFIG.get("WALK_FORWARD", {})
        test_start, test_end = wf.get(
            "TEST", (best_config.get("START_DATE"), best_config.get("END_DATE"))
        )
        ap.CONFIG.clear()
        ap.CONFIG.update(copy.deepcopy(best_config))
        ap.CONFIG["START_DATE"] = test_start
        ap.CONFIG["END_DATE"] = test_end
        _, _, test_metrics = ap.run_backtest(
            save_outputs=False,
            silent=True,
            use_regime_filter=best_config.get("USE_REGIME_FILTER"),
            regime_tickers=(
                best_config.get("REGIME_SELECTED_TICKERS")
                if best_config.get("USE_REGIME_FILTER")
                else None
            ),
            regime_requirement=best_config.get("REGIME_SELECTED_REQUIREMENT"),
        )
        test_cagr = float(test_metrics.get("CAGR", 0.0))
        test_mdd = abs(float(test_metrics.get("MDD", 0.0)))
        test_calmar = test_cagr / test_mdd if test_mdd > 0 else 0.0
        print("================= OUT-OF-SAMPLE TEST =================")
        print(f"Period: {test_start} ~ {test_end}")
        print(f"CAGR: {test_cagr:.2f}%")
        print(f"MDD : {-test_mdd:.2f}%")
        print(f"Win Rate: {float(test_metrics.get('WinRate', 0.0)):.2f}%")
        print(f"Calmar: {test_calmar:.2f}")
        print(f"Trades: {int(test_metrics.get('Trades', 0))}")
        print("=====================================================")
    else:
        print(f"[OPTUNA] Best unchanged: {existing_best_value} >= {study.best_value}")

    if best_value is not None and best_params is not None:
        print("[OPTUNA] Best value:", best_value)
        print("[OPTUNA] Best params:")
        for k, v in best_params.items():
            print(f"  - {k}: {v}")
    else:
        print("[OPTUNA] Best value: n/a (no completed trials)")
    print("[OPTUNA] Saved:", run_dir)

    if should_update_best:
        print("[OPTUNA] Best updated")


if __name__ == "__main__":
    main()
