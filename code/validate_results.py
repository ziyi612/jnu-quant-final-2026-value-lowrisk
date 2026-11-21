from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    root = parser.parse_args().results
    daily = pd.read_csv(root / "daily_portfolios.csv", parse_dates=["date"])
    metrics = pd.read_csv(root / "performance_metrics.csv")
    benchmark_path = root / "benchmark_csi300.csv"
    benchmark = pd.read_csv(benchmark_path, parse_dates=["date"]).set_index("date")
    checks = []
    for strategy, d in daily.groupby("strategy", sort=False):
        d = d.sort_values("date")
        nav = d.nav_net.to_numpy(); initial = np.r_[1., nav[:-1]]
        r = nav / initial - 1
        err = float(np.max(np.abs(r - d.return_net)))
        assert err < 1e-9
        capital_error = float(np.max(np.abs(d.return_gross - d.return_net - d.transaction_cost_fraction_previous_nav)))
        assert capital_error < 1e-10
        full = metrics[(metrics.strategy == strategy) & (metrics.period == "full") & (metrics.return_type == "net")].iloc[0]
        cagr = np.expm1(np.log(nav[-1]) * 252 / len(nav))
        vol = np.std(r, ddof=1) * np.sqrt(252)
        peak = np.maximum.accumulate(np.r_[1., nav])[1:]
        dd = nav / peak - 1
        assert abs(cagr - full.annualized_return) < 1e-9
        assert abs(vol - full.annualized_volatility) < 1e-9
        assert abs(dd.min() - full.max_drawdown) < 1e-9
        checks.append({"strategy": strategy, "net_return_reconciliation_max_error": err,
                       "daily_cost_identity_max_error": capital_error,
                       "cagr_independent_error": float(abs(cagr - full.annualized_return)), "passed": True})

    rows = pd.read_csv(root / "returns_monthly.csv")
    rows["month"] = pd.PeriodIndex(rows.month, freq="M")
    wide = rows[rows.strategy.isin(["baseline", "research"])].pivot(index="month", columns="strategy", values="return_net")
    diff = wide.research - wide.baseline
    stats = []
    for period, selection in [("full", diff), ("front_2018_2020", diff[diff.index.year <= 2020]),
                              ("historical_holdout_2021_2025", diff[diff.index.year >= 2021])]:
        fit = OLS(selection.to_numpy(), np.ones((len(selection), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": False})
        mean, se = float(fit.params[0]), float(fit.bse[0]); t = mean / se
        stats.append({"period": period, "months": len(selection), "mean_monthly_research_minus_baseline": mean,
                      "hac3_standard_error": se, "hac3_t_normal_approximation": t,
                      "normal_approximation_two_sided_p": float(2 * norm.sf(abs(t))),
                      "mean_ci95_low": mean - 1.95996398454 * se, "mean_ci95_high": mean + 1.95996398454 * se,
                      "qualification": "Paired calendar-month net returns; normal HAC(3) approximation; not a causal estimate or a prospective test."})
    pd.DataFrame(stats).to_csv(root / "paired_model_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

    risk = []
    for strategy in ["baseline", "research", "benchmark"]:
        if strategy == "benchmark":
            dates = benchmark.index; nav = benchmark.nav.to_numpy(); r = benchmark["return"].to_numpy()
        else:
            d = daily[daily.strategy == strategy].sort_values("date")
            dates = pd.DatetimeIndex(d.date); nav = d.nav_net.to_numpy(); r = d.return_net.to_numpy()
        b = benchmark.reindex(dates)["return"].to_numpy()
        fit = OLS(r, np.column_stack([np.ones(len(b)), b])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        augmented = np.r_[1., nav]
        dd = augmented / np.maximum.accumulate(augmented) - 1
        trough = int(dd.argmin()); peak = int(augmented[:trough+1].argmax())
        recovered = np.flatnonzero(augmented[trough+1:] >= augmented[peak])
        recovery = (trough + 1 + int(recovered[0])) if len(recovered) else None
        def date_label(i):
            return "initial_capital" if i == 0 else str(dates[i-1].date())
        risk.append({"strategy": strategy, "market_beta": float(fit.params[1]),
                     "market_return_correlation": float(np.corrcoef(r, b)[0, 1]),
                     "daily_market_regression_intercept": float(fit.params[0]),
                     "intercept_hac5_t": float(fit.tvalues[0]),
                     "maximum_drawdown": float(dd.min()), "drawdown_peak_date": date_label(peak),
                     "drawdown_trough_date": date_label(trough),
                     "recovery_date": date_label(recovery) if recovery is not None else "not_recovered_by_2025_12_31",
                     "trading_observations_peak_to_trough": trough - peak,
                     "qualification": "Market-only descriptive regression, not full factor alpha; CSI300 is a price index."})
    pd.DataFrame(risk).to_csv(root / "risk_episodes_and_beta.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    result = {"all_checks_passed": True, "independent_check_count": len(checks), "checks": checks,
              "paired_comparisons": stats, "risk_episodes": risk,
              "limits": ["Reconciles exported results; does not certify historical data vintages or actual exchange fills.",
                         "HAC intervals describe this fixed historical comparison; model family and signal choices were not preregistered before the historical data existed."]}
    (root / "independent_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"checked_strategies": len(checks), "paired_comparison": stats[0], "risk_episodes": risk}, ensure_ascii=False))


if __name__ == "__main__":
    main()
