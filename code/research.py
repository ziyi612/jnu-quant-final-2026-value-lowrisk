# Course reference: https://github.com/styluck/quant
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

FACTORS = ["value_bp", "small_size", "momentum_12_1", "low_volatility"]


def log(message: str) -> None:
    print(message, flush=True)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Bundle:
    frames: dict[str, pd.DataFrame]
    dates: pd.DatetimeIndex
    codes: np.ndarray
    adjusted: pd.DataFrame
    marks: np.ndarray
    ipo: np.ndarray
    out: np.ndarray
    names: dict[str, str]
    benchmark: pd.Series
    manifest: list[dict]


@dataclass
class Signal:
    index: int
    date: pd.Timestamp
    execute_index: int | None
    eligible: np.ndarray
    z: np.ndarray
    adv: np.ndarray
    raw: np.ndarray


def load_bundle(cfg: dict) -> Bundle:
    root = Path(cfg["data_dir"])
    frames, manifest = {}, []
    for field in ["close", "adj_factor", "pb", "total_mv", "is_st", "vol", "amount"]:
        path = root / f"{field}_sh.csv"
        f = pd.read_csv(path, index_col=0, parse_dates=True)
        f = f.loc[cfg["warmup_start"]:cfg["backtest_end"]].astype(float)
        if not f.index.is_unique or not f.columns.is_unique or not f.index.is_monotonic_increasing:
            raise ValueError(f"Nonunique or unsorted keys: {path}")
        if frames:
            base = frames["close"]
            if not f.index.equals(base.index) or not f.columns.equals(base.columns):
                raise ValueError(f"Date/security alignment mismatch: {path}")
        frames[field] = f
        manifest.append({"field": field, "path": path.name, "bytes": path.stat().st_size,
                         "sha256": sha256(path), "rows": len(f), "columns": len(f.columns),
                         "first_date": str(f.index[0].date()), "last_date": str(f.index[-1].date())})
    close = frames["close"]
    dates, codes = close.index, close.columns.to_numpy(dtype=str)
    basic_path = root / "sse_stock_basic.csv"
    basic = pd.read_csv(basic_path, dtype={"code": str})
    basic["ts_code"] = basic["code"].str.replace("sh.", "", regex=False) + ".SH"
    if basic.ts_code.duplicated().any():
        raise ValueError("Duplicate security metadata")
    basic = basic.set_index("ts_code").reindex(codes)
    ipo = pd.to_datetime(basic.ipoDate, errors="coerce").to_numpy(dtype="datetime64[D]")
    out = pd.to_datetime(basic.outDate, errors="coerce").to_numpy(dtype="datetime64[D]")
    names = basic.code_name.fillna("").to_dict()
    adjusted = (close * frames["adj_factor"]).where(close.gt(0) & frames["adj_factor"].gt(0))
    adjusted = adjusted.ffill()
    marks = np.nan_to_num(adjusted.to_numpy(), nan=0.0)
    bp = Path(cfg["benchmark_path"])
    bm = pd.read_csv(bp)
    if "code" in bm and not set(bm.code.dropna().astype(str)).issubset({"sh.000300", "000300.SH", "000300"}):
        raise ValueError("Benchmark code is not the supplied CSI300 series")
    benchmark = pd.Series(pd.to_numeric(bm.close, errors="raise").to_numpy(),
                          index=pd.to_datetime(bm.date), name="benchmark_close")
    if not benchmark.index.is_unique:
        raise ValueError("Duplicate benchmark dates")
    benchmark = benchmark.reindex(dates)
    if benchmark.isna().any():
        raise ValueError("Missing benchmark observations; no silent filling")
    for path, kind in [(basic_path, "security_metadata"), (bp, "CSI300_price_index")]:
        manifest.append({"field": kind, "path": path.name, "bytes": path.stat().st_size,
                         "sha256": sha256(path)})
    return Bundle(frames, dates, codes, adjusted, marks, ipo, out, names, benchmark, manifest)


def preprocess(raw: np.ndarray, eligible: np.ndarray, cfg: dict) -> np.ndarray:
    z = np.full_like(raw, np.nan, dtype=float)
    if not eligible.any():
        return z
    x = raw[eligible]
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    width = cfg["mad_multiplier"] * cfg["mad_consistency_factor"] * mad
    clipped = np.clip(x, median - width, median + width)
    std = clipped.std(axis=0, ddof=cfg["zscore_ddof"])
    z[eligible] = np.divide(clipped - clipped.mean(axis=0), std,
                            out=np.zeros_like(clipped), where=std > 1e-14)
    return z


def signal_from_raw(bundle: Bundle, i: int, raw: np.ndarray, adv: np.ndarray,
                    history_count: np.ndarray, cfg: dict) -> Signal:
    f, date = bundle.frames, bundle.dates[i]
    close, af = f["close"].iloc[i].to_numpy(), f["adj_factor"].iloc[i].to_numpy()
    pb, mv = f["pb"].iloc[i].to_numpy(), f["total_mv"].iloc[i].to_numpy()
    age = (date.to_datetime64().astype("datetime64[D]") - bundle.ipo).astype("timedelta64[D]").astype(float)
    ended = ~np.isnat(bundle.out) & (bundle.out <= np.datetime64(date.date()))
    eligible = (np.isfinite(raw).all(axis=1) & np.isfinite(close) & (close > 0)
                & np.isfinite(af) & (af > 0) & np.isfinite(pb) & (pb > 0)
                & np.isfinite(mv) & (mv > 0) & (f["is_st"].iloc[i].to_numpy() == 0)
                & (f["vol"].iloc[i].to_numpy() > 0) & np.isfinite(adv)
                & (adv >= cfg["minimum_adv_thousand_rmb"])
                & (history_count >= cfg["history_window"])
                & (age >= cfg["min_listing_calendar_days"]) & ~ended)
    directions = np.array([cfg["factor_directions"][factor] for factor in FACTORS], dtype=float)
    raw = raw * directions
    z = preprocess(raw, eligible, cfg)
    return Signal(i, date, i + 1 if i + 1 < len(bundle.dates) else None,
                  eligible, z, adv, raw)


def build_signals(bundle: Bundle, cfg: dict) -> list[Signal]:
    f, p = bundle.frames, bundle.adjusted
    ret = p.pct_change(fill_method=None)
    momentum = p.shift(cfg["momentum_short_skip"]) / p.shift(cfg["history_window"]) - 1
    risk = -ret.rolling(cfg["volatility_window"], min_periods=cfg["volatility_min_observations"]).std(ddof=1)
    adv = f["amount"].rolling(cfg["adv_window"], min_periods=cfg["adv_min_observations"]).mean()
    history = f["close"].rolling(cfg["history_window"], min_periods=1).count()
    idx = bundle.dates
    month_ends = np.flatnonzero(np.r_[idx.to_period("M")[:-1] != idx.to_period("M")[1:], True])
    first_signal_month = (pd.Timestamp(cfg["backtest_start"]) - pd.offsets.MonthEnd(1)).to_period("M")
    result = []
    for i in month_ends:
        if idx[i].to_period("M") < first_signal_month:
            continue
        pb, mv = f["pb"].iloc[i].to_numpy(), f["total_mv"].iloc[i].to_numpy()
        raw = np.column_stack([np.divide(1.0, pb, out=np.full_like(pb, np.nan), where=pb > 0),
                               -np.log(np.where(mv > 0, mv, np.nan)),
                               momentum.iloc[i].to_numpy(), risk.iloc[i].to_numpy()])
        result.append(signal_from_raw(bundle, int(i), raw, adv.iloc[i].to_numpy(),
                                      history.iloc[i].to_numpy(), cfg))
    return result


def independent_causal_signal(bundle: Bundle, i: int, cfg: dict,
                              perturb_future: bool = False) -> Signal:
    frames = {}
    for field, frame in bundle.frames.items():
        values = frame.to_numpy(copy=True)
        if perturb_future:
            values[i + 1:] = (values[i + 1:] * 7 + 123) if field != "is_st" else 1
        frames[field] = values[:i + 1]
    ap = pd.DataFrame(frames["close"] * frames["adj_factor"]).ffill().to_numpy()
    pb, mv = frames["pb"][-1], frames["total_mv"][-1]
    daily = np.divide(ap[1:], ap[:-1], out=np.full_like(ap[1:], np.nan), where=ap[:-1] > 0) - 1
    lastret = daily[-cfg["volatility_window"]:]
    cnt = np.isfinite(lastret).sum(axis=0)
    enough = cnt >= cfg["volatility_min_observations"]
    risk = np.full(lastret.shape[1], np.nan)
    risk[enough] = -np.nanstd(lastret[:, enough], axis=0, ddof=1)
    momentum = ap[-1 - cfg["momentum_short_skip"]] / ap[-1 - cfg["history_window"]] - 1
    amounts = frames["amount"][-cfg["adv_window"]:]
    adv = np.mean(amounts, axis=0)  # any missing observation invalidates a 20-day ADV
    history = np.isfinite(frames["close"][-cfg["history_window"]:]).sum(axis=0)
    raw = np.column_stack([1 / pb, -np.log(mv), momentum, risk])
    return signal_from_raw(bundle, i, raw, adv, history, cfg)


def ranked_selection(score: np.ndarray, eligible: np.ndarray, held: np.ndarray,
                     n: int, buffer_multiple: float) -> np.ndarray:
    candidates = np.flatnonzero(eligible)
    order = candidates[np.argsort(-score[candidates], kind="stable")]
    ranks = np.full(len(score), np.inf)
    ranks[order] = np.arange(1, len(order) + 1)
    retained = order[held[order] & (ranks[order] <= math.ceil(n * buffer_multiple))]
    if len(retained) > n:
        retained = retained[:n]
    fill = order[~np.isin(order, retained)][:max(0, n - len(retained))]
    return np.r_[retained, fill].astype(int)


def rebalance(units: np.ndarray, cash: float, marks: np.ndarray, selected: np.ndarray,
              can_buy: np.ndarray, can_sell: np.ndarray, cost: float) -> tuple:
    """Trade against equal pretrade-NAV dollar targets, freezing blocked positions."""
    values = units * marks
    nav = float(cash + values.sum())
    target = np.zeros_like(values)
    if len(selected):
        target[selected] = nav / len(selected)
    sells = np.where(can_sell, np.maximum(values - target, 0), 0)
    units = units.copy()
    units -= np.divide(sells, marks, out=np.zeros_like(sells), where=marks > 0)
    cash += float(sells.sum() * (1 - cost))
    aftersell = units * marks
    requests = np.where(can_buy, np.maximum(target - aftersell, 0), 0)
    required = float(requests.sum() * (1 + cost))
    scale = min(1.0, max(0.0, cash) / required) if required > 0 else 0.0
    buys = requests * scale
    units += np.divide(buys, marks, out=np.zeros_like(buys), where=marks > 0)
    cash -= float(buys.sum() * (1 + cost))
    fees = float((sells.sum() + buys.sum()) * cost)
    if abs(cash) < 1e-13:
        cash = 0.0
    if cash < -1e-10 or units.min() < -1e-10:
        raise AssertionError("Negative cash or units")
    units[abs(units) < 1e-15] = 0
    return units, cash, buys, sells, fees


def trade_masks(bundle: Bundle, i: int, cfg: dict, terminated: np.ndarray | None = None) -> tuple:
    close = bundle.frames["close"].iloc[i].to_numpy()
    vol = bundle.frames["vol"].iloc[i].to_numpy()
    can = np.isfinite(close) & (close > 0) & np.isfinite(vol) & (vol > 0)
    if terminated is not None:
        can &= ~terminated
    previous = bundle.frames["close"].iloc[:i].ffill().iloc[-1].to_numpy() if i else close
    delta = np.divide(close, previous, out=np.full_like(close, np.nan), where=previous > 0) - 1
    limit = np.where(np.char.startswith(bundle.codes, "688") | np.char.startswith(bundle.codes, "689"),
                     cfg["star_limit_proxy"], cfg["mainboard_limit_proxy"])
    limit = np.where(bundle.frames["is_st"].iloc[i].to_numpy() == 1, cfg["st_limit_proxy"], limit)
    if cfg["approximate_price_limit_filter"]:
        return can & (delta < limit), can & (delta > -limit)
    return can.copy(), can.copy()


def make_specs(cfg: dict) -> list[dict]:
    baseline = np.array([cfg["baseline_factor_weights"].get(factor, 0) for factor in FACTORS], dtype=float)
    research = np.array([cfg["research_factor_weights"].get(factor, 0) for factor in FACTORS], dtype=float)
    for weights in [baseline, research]:
        if (weights < 0).any() or not np.isclose(weights.sum(), 1):
            raise ValueError("Factor weights must be nonnegative and sum to one")
    def spec(name, weights, n=None, buffer=None, bps=None, terminal="zero"):
        return {"strategy": name, "weights": weights,
                "holdings": cfg["main_holdings"] if n is None else n,
                "buffer": cfg["buffer_multiple"] if buffer is None else buffer,
                "cost_bps": cfg["one_way_cost_bps"] if bps is None else bps, "terminal": terminal}
    without_momentum, without_lowvol = research.copy(), research.copy()
    without_momentum[2], without_lowvol[3] = 0, 0
    without_momentum /= without_momentum.sum()
    without_lowvol /= without_lowvol.sum()
    result = [spec("baseline", baseline), spec("research", research),
              spec("research_no_buffer", research, buffer=1.0),
              spec("research_no_momentum", without_momentum),
              spec("research_no_lowvol", without_lowvol)]
    for n in cfg["holdings_scenarios"]:
        result.append(spec(f"research_n{n}", research, n=n))
    for bps in cfg["cost_scenarios_bps"]:
        for model, weights in [("baseline", baseline), ("research", research)]:
            result.append(spec(f"{model}_cost{bps}bp", weights, bps=bps))
    for model, weights in [("baseline", baseline), ("research", research)]:
        result.append(spec(f"{model}_terminal_upper", weights, terminal="last_mark_cash"))
    return result


def simulate(bundle: Bundle, signals: list[Signal], spec: dict, cfg: dict,
             all_trade_masks: dict[int, tuple]) -> dict:
    start = int(bundle.dates.searchsorted(pd.Timestamp(cfg["backtest_start"])))
    end = int(bundle.dates.searchsorted(pd.Timestamp(cfg["backtest_end"]), side="right"))
    schedule = {s.execute_index: s for s in signals if s.execute_index is not None
                and start <= s.execute_index < end}
    units = np.zeros(len(bundle.codes))
    cash, previous_nav, gross_nav = 1.0, 1.0, 1.0
    terminated = np.zeros(len(bundle.codes), dtype=bool)
    daily, holdings, trades, rebalances, events = [], [], [], [], []
    max_weight_error = 0.0
    min_cash, min_units = 1.0, 0.0
    for i in range(start, end):
        date, marks = bundle.dates[i], bundle.marks[i]
        due = ~np.isnat(bundle.out) & (bundle.out <= np.datetime64(date.date())) & ~terminated
        for j in np.flatnonzero(due & (units > 0)):
            value = float(units[j] * marks[j])
            recovery = value if spec["terminal"] == "last_mark_cash" else 0.0
            cash += recovery
            units[j] = 0
            events.append({"strategy": spec["strategy"], "date": date, "code": bundle.codes[j],
                           "recorded_out_date": str(bundle.out[j]), "mark_value_nav_units": value,
                           "recovery_nav_units": recovery, "assumed_loss_nav_units": value - recovery,
                           "loss_fraction_previous_nav": (value - recovery) / previous_nav,
                           "assumption": spec["terminal"]})
        terminated |= due
        pre_nav = float(cash + np.dot(units, marks))
        gross_return = pre_nav / previous_nav - 1
        gross_nav *= 1 + gross_return
        fee = bought = sold = 0.0
        if i in schedule:
            signal = schedule[i]
            scores = np.nan_to_num(signal.z, nan=0) @ spec["weights"]
            selected = ranked_selection(scores, signal.eligible, units > 0,
                                        spec["holdings"], spec["buffer"])
            buyable, sellable = all_trade_masks[i]
            buyable, sellable = buyable & ~terminated, sellable & ~terminated
            before_units = units.copy()
            units, cash, buys, sells, fee = rebalance(units, cash, marks, selected,
                                                     buyable, sellable, spec["cost_bps"] / 10000)
            bought, sold = float(buys.sum()), float(sells.sum())
            nav = float(cash + np.dot(units, marks))
            weights = units * marks / nav
            changes = (buys + sells) / pre_nav
            traded_idx = np.flatnonzero(changes > 1e-12)
            valid_capacity = traded_idx[np.isfinite(signal.adv[traded_idx]) & (signal.adv[traded_idx] > 0)]
            capacity = (signal.adv[valid_capacity] * 1000 * cfg["capacity_participation"]
                        * cfg["capacity_execution_days"] / changes[valid_capacity])
            target_flags = np.zeros(len(units), dtype=bool)
            target_flags[selected] = True
            for j in np.flatnonzero(units > 0):
                holdings.append({"strategy": spec["strategy"], "signal_date": signal.date,
                                 "execution_date": date, "code": bundle.codes[j], "weight": weights[j],
                                 "adjusted_units": units[j], "adjusted_mark": marks[j],
                                 "target_selected": bool(target_flags[j]),
                                 "signal_eligible": bool(signal.eligible[j]), "score": scores[j],
                                 **{factor: signal.z[j, k] for k, factor in enumerate(FACTORS)}})
            for j in traded_idx:
                trades.append({"strategy": spec["strategy"], "signal_date": signal.date,
                               "execution_date": date, "code": bundle.codes[j],
                               "buy_nav_units": buys[j], "sell_nav_units": sells[j],
                               "absolute_trade_weight": changes[j], "adv_thousand_rmb": signal.adv[j],
                               "capacity_rmb": (signal.adv[j] * 1000 * cfg["capacity_participation"]
                                                * cfg["capacity_execution_days"] / changes[j])
                               if np.isfinite(signal.adv[j]) else np.nan})
            exposures = weights @ np.nan_to_num(signal.z, nan=0)
            blocked_held = (before_units > 0) & ~sellable
            rebalances.append({"strategy": spec["strategy"], "signal_date": signal.date,
                               "execution_date": date, "eligible_count": int(signal.eligible.sum()),
                               "target_count": len(selected), "held_count": int((units > 0).sum()),
                               "cash_weight": cash / nav, "invested_weight": float(weights.sum()),
                               "unmodeled_weight": float(weights[~signal.eligible].sum()),
                               "blocked_sell_held_count": int(blocked_held.sum()),
                               "blocked_target_buy_count": int((target_flags & ~buyable).sum()),
                               "buy_weight_pretrade_nav": bought / pre_nav,
                               "sell_weight_pretrade_nav": sold / pre_nav,
                               "one_way_turnover": (bought + sold) / (2 * pre_nav),
                               "transaction_cost_fraction_pretrade_nav": fee / pre_nav,
                               "capacity_min_rmb": float(capacity.min()) if len(capacity) else np.nan,
                               "capacity_p10_rmb": float(np.quantile(capacity, .1)) if len(capacity) else np.nan,
                               "capacity_observed_traded_count": len(capacity),
                               "capacity_missing_traded_count": len(traded_idx) - len(capacity),
                               **{factor: exposures[k] for k, factor in enumerate(FACTORS)}})
        nav = float(cash + np.dot(units, marks))
        if nav <= 0:
            raise AssertionError("Nonpositive portfolio value")
        weight_sum = float(np.dot(units, marks) / nav + cash / nav)
        max_weight_error = max(max_weight_error, abs(weight_sum - 1))
        min_cash, min_units = min(min_cash, cash), min(min_units, float(units.min()))
        if cash < -1e-10 or units.min() < -1e-10 or not np.isfinite(nav):
            raise AssertionError("Nonfinite or negative portfolio assets")
        daily.append({"strategy": spec["strategy"], "date": date, "nav_net": nav,
                      "nav_gross": gross_nav, "return_net": nav / previous_nav - 1,
                      "return_gross": gross_return, "cash_weight": cash / nav,
                      "invested_weight": 1 - cash / nav, "holdings_count": int((units > 0).sum()),
                      "transaction_cost_nav_units": fee,
                      "transaction_cost_fraction_previous_nav": fee / previous_nav,
                      "buy_nav_units": bought, "sell_nav_units": sold,
                      "weight_cash_sum": weight_sum})
        previous_nav = nav
    return {"spec": spec, "daily": pd.DataFrame(daily), "holdings": pd.DataFrame(holdings),
            "trades": pd.DataFrame(trades), "rebalances": pd.DataFrame(rebalances),
            "terminal_events": pd.DataFrame(events), "last_units": units, "last_cash": cash,
            "validation": {"strategy": spec["strategy"], "maximum_weight_cash_sum_error": max_weight_error,
                           "minimum_cash": min_cash, "minimum_units": min_units}}


def one_period_labels(bundle: Bundle, signal: Signal, end_idx: int, cfg: dict,
                      masks: tuple, terminal: str = "zero") -> tuple:
    e = signal.execute_index
    if e is None:
        raise ValueError("Signal has no execution observation")
    start_mark, end_mark = bundle.marks[e], bundle.marks[end_idx]
    already_ended = ~np.isnat(bundle.out) & (bundle.out <= np.datetime64(bundle.dates[e].date()))
    can_buy = masks[0] & ~already_ended
    y = np.divide(end_mark, start_mark, out=np.ones_like(start_mark), where=start_mark > 0) - 1
    exited = (~np.isnat(bundle.out) & (bundle.out > np.datetime64(bundle.dates[e].date()))
              & (bundle.out <= np.datetime64(bundle.dates[end_idx].date())))
    if terminal == "zero":
        y[exited] = -1
    y[~can_buy] = 0.0  # this equal-weight cohort leaves blocked initial purchases in cash
    return y, exited & can_buy, ~can_buy


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-14 or np.std(b) < 1e-14:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def factor_diagnostics(bundle: Bundle, signals: list[Signal], cfg: dict,
                       masks: dict[int, tuple]) -> dict:
    ic_rows, group_rows, regressions, coefficient_rows, factor_corr = [], [], {}, [], []
    executable = [s for s in signals if s.execute_index is not None]
    for k, signal in enumerate(executable):
        e = signal.execute_index
        next_idx = executable[k + 1].execute_index if k + 1 < len(executable) else len(bundle.dates) - 1
        partial = k + 1 == len(executable)
        ids = np.flatnonzero(signal.eligible)
        if len(ids) < 20:
            continue
        y, exits, blocked = one_period_labels(bundle, signal, next_idx, cfg, masks[e], "zero")
        yu, _, _ = one_period_labels(bundle, signal, next_idx, cfg, masks[e], "last_mark_cash")
        z, yy = signal.z[ids], y[ids]
        design = np.column_stack([np.ones(len(ids)), z])
        coefficients, _, rank, _ = np.linalg.lstsq(design, yy, rcond=None)
        fitted = design @ coefficients
        ss = float(np.square(yy - yy.mean()).sum())
        r2 = 1 - float(np.square(yy - fitted).sum()) / ss if ss > 0 else np.nan
        row = {"signal_date": signal.date, "execution_date": bundle.dates[e],
               "end_execution_or_dataset_date": bundle.dates[next_idx], "partial_final_period": partial,
               "sample_count": len(ids), "regression_rank": int(rank), "r_squared": r2,
               "intercept": coefficients[0], "terminal_proxy_count": int(exits[ids].sum()),
               "blocked_entry_count": int(blocked[ids].sum()),
               **{f: coefficients[j + 1] for j, f in enumerate(FACTORS)}}
        coefficient_rows.append(row)
        regressions[signal.date] = {"coefficients": coefficients, "start_idx": e,
                                    "end_idx": next_idx, "partial": partial}
        for j, factor in enumerate(FACTORS):
            ric = corr(rankdata(z[:, j]), rankdata(yy))
            ic_rows.append({"factor": factor, "signal_date": signal.date,
                            "execution_date": bundle.dates[e], "end_date": bundle.dates[next_idx],
                            "partial_final_period": partial, "n": len(ids),
                            "rank_ic": ric, "pearson_ic": corr(z[:, j], yy),
                            "rank_ic_terminal_upper": corr(rankdata(z[:, j]), rankdata(yu[ids])),
                            "terminal_proxy_count": int(exits[ids].sum()),
                            "blocked_entry_cash_count": int(blocked[ids].sum())})
            order = ids[np.argsort(signal.z[ids, j], kind="stable")]
            for g, group in enumerate(np.array_split(order, 5), start=1):
                group_rows.append({"factor": factor, "signal_date": signal.date,
                                   "execution_date": bundle.dates[e], "end_date": bundle.dates[next_idx],
                                   "partial_final_period": partial, "group": g, "n": len(group),
                                   "return_gross": float(y[group].mean()),
                                   "return_terminal_upper": float(yu[group].mean()),
                                   "terminal_proxy_count": int(exits[group].sum()),
                                   "blocked_entry_cash_count": int(blocked[group].sum())})
            for jj in range(j + 1, len(FACTORS)):
                factor_corr.append({"signal_date": signal.date, "factor_a": factor,
                                    "factor_b": FACTORS[jj], "correlation": corr(z[:, j], z[:, jj])})
    return {"ic": pd.DataFrame(ic_rows), "groups": pd.DataFrame(group_rows),
            "coefficients": pd.DataFrame(coefficient_rows), "regressions": regressions,
            "cross_correlations": pd.DataFrame(factor_corr)}


def nw_mean_t(values: np.ndarray, lag: int = 3) -> float:
    x = values[np.isfinite(values)]
    n = len(x)
    if n < 5:
        return np.nan
    u = x - x.mean()
    lrv = np.dot(u, u) / n
    for j in range(1, min(lag, n - 1) + 1):
        lrv += 2 * (1 - j / (lag + 1)) * np.dot(u[j:], u[:-j]) / n
    return float(x.mean() / np.sqrt(lrv / n)) if lrv > 0 else np.nan


def summarize_ic(frame: pd.DataFrame) -> pd.DataFrame:
    result = []
    for factor, group in frame.groupby("factor", sort=False):
        for period, lo, hi in [("full", "2017-12-01", "2025-12-31"),
                               ("front_2018_2020", "2017-12-01", "2020-11-30"),
                               ("historical_holdout_2021_2025", "2020-12-01", "2025-12-31")]:
            g = group[group.signal_date.between(lo, hi) & ~group.partial_final_period]
            x = g.rank_ic.to_numpy()
            std = np.std(x, ddof=1) if len(x) > 1 else np.nan
            result.append({"factor": factor, "period": period, "complete_periods": len(x),
                           "mean_rank_ic": x.mean() if len(x) else np.nan,
                           "std_rank_ic": std, "icir_monthly": x.mean()/std if std > 0 else np.nan,
                           "icir_annualized": x.mean()/std*np.sqrt(12) if std > 0 else np.nan,
                           "positive_ic_fraction": float((x > 0).mean()) if len(x) else np.nan,
                           "newey_west_mean_t_lag3": nw_mean_t(x),
                           "mean_rank_ic_terminal_upper": g.rank_ic_terminal_upper.mean()})
    return pd.DataFrame(result)


def attribution(run: dict, diagnostics: dict, bundle: Bundle) -> pd.DataFrame:
    rows = []
    daily = run["daily"].set_index("date")
    for _, reb in run["rebalances"].iterrows():
        key = reb.signal_date
        if key not in diagnostics["regressions"]:
            continue
        reg = diagnostics["regressions"][key]
        lo, hi = bundle.dates[reg["start_idx"]], bundle.dates[reg["end_idx"]]
        interval = daily.loc[(daily.index > lo) & (daily.index <= hi)]
        gross_return = float(np.prod(1 + interval.return_gross) - 1)
        net_return = float(np.prod(1 + interval.return_net) - 1)
        c = reg["coefficients"]
        intercept = float(c[0] * reb.invested_weight)
        contributions = np.array([reb[factor] * c[j + 1] for j, factor in enumerate(FACTORS)])
        residual = gross_return - intercept - float(contributions.sum())
        closure = gross_return - (intercept + float(contributions.sum()) + residual)
        rows.append({"strategy": run["spec"]["strategy"], "signal_date": key,
                     "execution_date": lo, "end_date": hi,
                     "partial_final_period": reg["partial"], "gross_return": gross_return,
                     "net_return_excluding_entry_cost_including_end_rebalance_cost": net_return,
                     "intercept_contribution": intercept,
                     **{factor + "_contribution": contributions[j] for j, factor in enumerate(FACTORS)},
                     "residual_contribution": residual, "closure_error": closure,
                     "invested_weight": reb.invested_weight, "unmodeled_weight": reb.unmodeled_weight})
    return pd.DataFrame(rows)


def performance(returns: np.ndarray, benchmark: np.ndarray, annual: int = 252) -> dict:
    n = len(returns)
    if n == 0:
        return {}
    wealth, bmwealth = np.prod(1 + returns), np.prod(1 + benchmark)
    nav = np.r_[1.0, np.cumprod(1 + returns)]
    dd = nav / np.maximum.accumulate(nav) - 1
    vol = float(np.std(returns, ddof=1) * np.sqrt(annual))
    active = returns - benchmark
    te = float(np.std(active, ddof=1) * np.sqrt(annual))
    cagr, bmcagr = wealth ** (annual / n) - 1, bmwealth ** (annual / n) - 1
    return {"observations": n, "total_return": wealth - 1, "annualized_return": cagr,
            "annualized_volatility": vol, "sharpe_rf0": returns.mean()*annual/vol if vol > 0 else np.nan,
            "max_drawdown": float(dd.min()), "benchmark_total_return": bmwealth - 1,
            "benchmark_annualized_return": bmcagr, "annualized_return_minus_benchmark": cagr-bmcagr,
            "relative_wealth_annualized_excess": (wealth / bmwealth)**(annual/n)-1,
            "tracking_error": te, "information_ratio": active.mean()*annual/te if te > 0 else np.nan,
            "daily_win_fraction": float((returns > 0).mean())}


def build_period_metrics(run: dict, benchmark_returns: pd.Series, cfg: dict) -> pd.DataFrame:
    result = []
    d = run["daily"]
    for name, lo, hi in [("full", cfg["backtest_start"], cfg["backtest_end"]),
                         ("front_2018_2020", cfg["backtest_start"], cfg["development_end"]),
                         ("historical_holdout_2021_2025", cfg["holdout_start"], cfg["backtest_end"])]:
        part = d[d.date.between(lo, hi)]
        b = benchmark_returns.reindex(part.date).to_numpy()
        for kind in ["net", "gross"]:
            result.append({"strategy": run["spec"]["strategy"], "period": name,
                           "return_type": kind, "start": part.date.min(), "end": part.date.max(),
                           "cost_bps": run["spec"]["cost_bps"], "target_holdings": run["spec"]["holdings"],
                           "terminal_assumption": run["spec"]["terminal"],
                           **performance(part["return_"+kind].to_numpy(), b, cfg["annual_trading_days"])})
    return pd.DataFrame(result)


def unit_checks() -> list[dict]:
    units, cash, buys, sells, cost = rebalance(np.zeros(2), 1, np.ones(2), np.array([0, 1]),
                                             np.ones(2, bool), np.ones(2, bool), 0)
    assert np.allclose(units, [.5, .5]) and cash == 0 and cost == 0
    u2, c2, _, _, fee = rebalance(np.zeros(2), 1, np.ones(2), np.array([0, 1]),
                                 np.ones(2, bool), np.ones(2, bool), .01)
    assert np.isclose(u2.sum()+c2, 1/1.01) and np.isclose(fee, .01/1.01)
    u3, c3, _, _, fee3 = rebalance(u2, c2, np.ones(2), np.array([], dtype=int),
                                  np.ones(2, bool), np.ones(2, bool), .01)
    assert np.isclose(c3, (1-.01)/(1+.01)) and u3.sum() == 0
    frozen_u, frozen_c, _, _, _ = rebalance(np.array([.5, 0]), .5, np.ones(2), np.array([1]),
                                            np.ones(2, bool), np.array([False, True]), 0)
    assert np.allclose(frozen_u, [.5, .5]) and frozen_c == 0
    return [{"check": "constant_price_zero_cost_equal_allocation", "passed": True},
            {"check": "constant_price_exact_buy_and_sell_cost_accounting", "passed": True},
            {"check": "blocked_sale_preserves_units_and_cash_scales_new_purchase", "passed": True}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--benchmark-path", type=Path)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    if args.data_dir is None:
        cfg["data_dir"] = cfg_path.parent / cfg["data_dir"]
    else:
        cfg["data_dir"] = args.data_dir
    if args.benchmark_path is None:
        cfg["benchmark_path"] = cfg_path.parent / cfg["benchmark_path"]
    else:
        cfg["benchmark_path"] = args.benchmark_path
    started = time.time()
    log("Loading original daily tables and checking alignment...")
    bundle = load_bundle(cfg)
    write_json(output / "run_manifest.json", {"config_sha256": sha256(cfg_path),
                "research_script_sha256": sha256(Path(__file__)),
                "data": bundle.manifest, "python": sys.version, "pandas": pd.__version__,
                "numpy": np.__version__,
                "benchmark_epoch": "First strategy execution close; first benchmark return set to zero. Initial strategy construction cost retained.",
                "factor_label": "Executable equal-weight one-period cohort; signal-date buckets fixed before labels; blocked initial buys remain cash at zero return, with the stated terminal payoff proxy."})
    log("Calculating monthly factors...")
    signals = build_signals(bundle, cfg)
    valid_signals = [s for s in signals if s.execute_index is not None]
    masks = {s.execute_index: trade_masks(bundle, s.execute_index, cfg) for s in valid_signals}
    signal_rows, overview = [], []
    for signal in signals:
        overview.append({"signal_date": signal.date, "execution_date": bundle.dates[signal.execute_index]
                         if signal.execute_index is not None else pd.NaT,
                         "eligible_count": int(signal.eligible.sum()), "historical_signal_only": signal.execute_index is None})
        for j in np.flatnonzero(signal.eligible):
            signal_rows.append({"signal_date": signal.date, "code": bundle.codes[j],
                                "adv_thousand_rmb": signal.adv[j],
                                **{factor: signal.z[j, k] for k, factor in enumerate(FACTORS)},
                                **{factor+"_raw": signal.raw[j, k] for k, factor in enumerate(FACTORS)}})
    save_csv(pd.DataFrame(overview), output / "universe_by_signal.csv")
    save_csv(pd.DataFrame(signal_rows), output / "signal_factors.csv")
    del signal_rows
    diagnostics = factor_diagnostics(bundle, signals, cfg, masks)
    save_csv(diagnostics["ic"], output / "factor_ic_monthly.csv")
    save_csv(summarize_ic(diagnostics["ic"]), output / "factor_ic_summary.csv")
    save_csv(diagnostics["groups"], output / "factor_groups_monthly.csv")
    save_csv(diagnostics["coefficients"], output / "cross_sectional_regressions.csv")
    save_csv(diagnostics["cross_correlations"], output / "factor_cross_correlations.csv")
    runs, metrics, attribution_frames = {}, [], []
    benchmark_returns = bundle.benchmark.pct_change(fill_method=None)
    first_execution_index = min(s.execute_index for s in valid_signals)
    benchmark_returns.iloc[first_execution_index] = 0.0
    all_specs = make_specs(cfg)
    for spec in all_specs:
        log(f"Simulating {spec['strategy']} ({len(runs)+1}/{len(all_specs)})...")
        run = simulate(bundle, signals, spec, cfg, masks)
        runs[spec["strategy"]] = run
        metrics.append(build_period_metrics(run, benchmark_returns, cfg))
        attribution_frames.append(attribution(run, diagnostics, bundle))
    for field, filename in [("daily", "daily_portfolios.csv"), ("holdings", "holdings_monthly.csv"),
                            ("trades", "executed_trades.csv"), ("rebalances", "rebalances.csv"),
                            ("terminal_events", "terminal_events.csv")]:
        save_csv(pd.concat([run[field] for run in runs.values()], ignore_index=True), output / filename)
    metric_frame = pd.concat(metrics, ignore_index=True)
    save_csv(metric_frame, output / "performance_metrics.csv")
    attrs = pd.concat(attribution_frames, ignore_index=True)
    save_csv(attrs, output / "return_attribution.csv")
    all_daily = pd.concat([run["daily"] for run in runs.values()], ignore_index=True)
    month_rows, annual_rows = [], []
    for name, run in runs.items():
        for column, destination, frequency in [("month", month_rows, "M"), ("year", annual_rows, "Y")]:
            for period, frame in run["daily"].groupby(run["daily"].date.dt.to_period(frequency)):
                row = {"strategy": name, column: str(period), "return_net": np.prod(1+frame.return_net)-1,
                       "return_gross": np.prod(1+frame.return_gross)-1,
                       "benchmark_return": np.prod(1+benchmark_returns.reindex(frame.date))-1,
                       "mean_cash_weight": frame.cash_weight.mean(),
                       "transaction_cost_nav_units": frame.transaction_cost_nav_units.sum()}
                destination.append(row)
    save_csv(pd.DataFrame(month_rows), output / "returns_monthly.csv")
    save_csv(pd.DataFrame(annual_rows), output / "returns_annual.csv")
    start_date = all_daily.date.min()
    bm = pd.DataFrame({"date": bundle.dates, "close": bundle.benchmark.to_numpy(),
                       "return": benchmark_returns.to_numpy()})
    bm = bm[bm.date >= start_date].copy()
    bm["nav"] = (1 + bm["return"]).cumprod()
    save_csv(bm, output / "benchmark_csi300.csv")
    bm_metrics = []
    for period, lo, hi in [("full", cfg["backtest_start"], cfg["backtest_end"]),
                           ("front_2018_2020", cfg["backtest_start"], cfg["development_end"]),
                           ("historical_holdout_2021_2025", cfg["holdout_start"], cfg["backtest_end"])]:
        r = benchmark_returns.loc[lo:hi].to_numpy()
        bm_metrics.append({"period": period, **performance(r, r, cfg["annual_trading_days"])})
    save_csv(pd.DataFrame(bm_metrics), output / "benchmark_metrics.csv")
    style_rows = []
    for name, run in runs.items():
        for j, factor in enumerate(FACTORS):
            for jj in range(j+1, len(FACTORS)):
                style_rows.append({"strategy": name, "factor_a": factor, "factor_b": FACTORS[jj],
                                   "time_series_exposure_correlation": run["rebalances"][[factor, FACTORS[jj]]].corr().iloc[0, 1]})
    save_csv(pd.DataFrame(style_rows), output / "style_exposure_correlations.csv")
    latest = signals[-1]
    run = runs["research"]
    latest_weights = np.array([cfg["research_factor_weights"].get(factor, 0) for factor in FACTORS])
    score = np.nan_to_num(latest.z, nan=0) @ latest_weights
    selected = ranked_selection(score, latest.eligible, run["last_units"]>0, cfg["main_holdings"], cfg["buffer_multiple"])
    ordered = np.flatnonzero(latest.eligible)
    ordered = ordered[np.argsort(-score[ordered], kind="stable")]
    ranks = {int(j): k+1 for k, j in enumerate(ordered)}
    latest_rows = [{"signal_date": latest.date, "code": bundle.codes[j], "name": bundle.names[bundle.codes[j]],
                    "score_rank": ranks[j], "score": score[j], "target_weight": 1/len(selected),
                    "raw_close": bundle.frames["close"].iloc[latest.index, j],
                    "adv_rmb_assumed": latest.adv[j]*1000,
                    "historical_signal_requires_2026_refresh": True,
                    **{factor: latest.z[j, k] for k, factor in enumerate(FACTORS)}} for j in selected]
    save_csv(pd.DataFrame(latest_rows), output / "latest_historical_signal_20251231.csv")
    checks = unit_checks()
    test_signal = min(valid_signals, key=lambda s: abs((s.date-pd.Timestamp("2020-12-31")).days))
    a = independent_causal_signal(bundle, test_signal.index, cfg, False)
    b = independent_causal_signal(bundle, test_signal.index, cfg, True)
    assert np.array_equal(a.eligible, test_signal.eligible) and np.array_equal(a.eligible, b.eligible)
    assert np.allclose(a.z, test_signal.z, equal_nan=True, atol=1e-10)
    assert np.allclose(a.z, b.z, equal_nan=True, atol=1e-13)
    checks.append({"check": "future_perturbation_and_independent_trailing_window_signal_invariance",
                   "date": test_signal.date, "passed": True,
                   "maximum_standardized_factor_error": float(np.nanmax(abs(a.z-test_signal.z)))})
    closure = float(attrs.closure_error.abs().max())
    assert closure < 1e-12
    checks.append({"check": "all_attribution_rows_close_with_explicit_residual", "passed": True,
                   "rows": len(attrs), "max_absolute_closure_error": closure})
    for run in runs.values():
        assert run["validation"]["maximum_weight_cash_sum_error"] < 1e-12
        checks.append({"check": "daily_nonnegative_assets_and_weights_plus_cash", "passed": True,
                       **run["validation"]})
    write_json(output / "validation.json", {"checks": checks,
                 "limits": ["Checks test bookkeeping/causal alignment, not economic correctness of source revisions or executable auction fills.",
                            "Attribution residual absorbs model misspecification and different execution/holdings; closure is an accounting identity, not goodness of fit.",
                            "No 2026 stock history or live trading verification."]})
    fee_rows = []
    daily_fee = (cfg["management_fee_annual"]+cfg["custody_fee_annual"]) / cfg["annual_trading_days"]
    for name in ["baseline", "research"]:
        d = runs[name]["daily"]
        fr = (1+d.return_net.to_numpy())*(1-daily_fee)-1
        df = pd.DataFrame({"strategy": name, "date": d.date, "return_after_assumed_product_fees": fr,
                           "nav_after_assumed_product_fees": np.cumprod(1+fr),
                           "daily_assumed_product_fee_rate": daily_fee})
        fee_rows.append(df)
    save_csv(pd.concat(fee_rows, ignore_index=True), output / "daily_after_product_fees.csv")
    product_metrics = []
    for name, df in zip(["baseline", "research"], fee_rows):
        for period, lo, hi in [("full", cfg["backtest_start"], cfg["backtest_end"]),
                              ("front_2018_2020", cfg["backtest_start"], cfg["development_end"]),
                              ("historical_holdout_2021_2025", cfg["holdout_start"], cfg["backtest_end"])]:
            sub = df[df.date.between(lo, hi)]
            product_metrics.append({"strategy": name, "period": period,
                **performance(sub.return_after_assumed_product_fees.to_numpy(),
                              benchmark_returns.reindex(sub.date).to_numpy(), cfg["annual_trading_days"])})
    save_csv(pd.DataFrame(product_metrics), output / "product_fee_metrics.csv")
    summary = {"title": cfg["project_title"], "elapsed_seconds": time.time()-started,
               "strategies": len(runs), "trade_date_count": len(runs["research"]["daily"]),
               "signals_count_including_latest_unexecuted": len(signals),
               "minimum_signal_universe": min(int(s.eligible.sum()) for s in signals),
               "maximum_signal_universe": max(int(s.eligible.sum()) for s in signals),
               "main_net_metrics": metric_frame[(metric_frame.strategy.isin(["baseline", "research"]))
                     & (metric_frame.return_type=="net")].to_dict("records"),
               "main_terminal_events": {name: len(runs[name]["terminal_events"]) for name in ["baseline", "research"]},
               "all_checks_passed": True}
    write_json(output / "run_summary.json", summary)
    log(json.dumps(clean_json(summary), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
