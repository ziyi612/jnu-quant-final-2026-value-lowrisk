from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

NAMES = {"baseline": "价值与市值基准", "research": "四因子研究模型", "benchmark": "沪深300价格指数"}
FACTORS = {"value_bp": "价值 BP", "small_size": "小市值", "momentum_12_1": "中期动量", "low_volatility": "低波动"}
COLORS = {"baseline": "#54758d", "research": "#bd582e", "benchmark": "#768379"}


def configure() -> None:
    for candidate in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            family = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 17, "axes.labelsize": 13,
                         "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "savefig.dpi": 240,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    return pd.read_csv(path, encoding="utf-8-sig")


def pct(ax, axis="y", decimals=0):
    getattr(ax, axis + "axis").set_major_formatter(PercentFormatter(1, decimals=decimals))


def finish(fig, out: Path, stem: str, registry: list, title: str, source: list, note: str = ""):
    fig.tight_layout(pad=1.4)
    target = out / (stem + ".png")
    fig.savefig(target, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    registry.append({"id": stem, "title": title, "path": str(target.resolve()),
                     "source_files": source, "qualification": note})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "figures")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    configure()
    registry = []
    daily = read(a.results, "daily_portfolios.csv")
    daily["date"] = pd.to_datetime(daily.date)
    benchmark = read(a.results, "benchmark_csi300.csv")
    benchmark["date"] = pd.to_datetime(benchmark.date)
    metrics = read(a.results, "performance_metrics.csv")
    full = metrics[(metrics.period == "full") & (metrics.return_type == "net")].set_index("strategy")
    bm_metrics = read(a.results, "benchmark_metrics.csv").set_index("period")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for key in ["baseline", "research"]:
        d = daily[daily.strategy == key]
        ax.plot(d.date, d.nav_net, label=NAMES[key], color=COLORS[key], linewidth=1.7)
    ax.plot(benchmark.date, benchmark.nav, label=NAMES["benchmark"], color=COLORS["benchmark"], linestyle="--", linewidth=1.5)
    ax.set(title="同一执行口径下的累计净值", ylabel="初始资产 = 1")
    ax.set_xlim(daily.date.min(), daily.date.max())
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=.16)
    finish(fig, a.output, "01_nav", registry, "累计净值", ["daily_portfolios.csv", "benchmark_csi300.csv"],
           "股票采用复权收益，沪深300为价格指数；策略净值已扣每边15bp综合交易成本，未扣产品费率。")

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for key in ["baseline", "research"]:
        d = daily[daily.strategy == key]
        nav = d.nav_net.to_numpy()
        dd = nav / np.maximum.accumulate(np.r_[1, nav])[1:] - 1
        ax.plot(d.date, dd, label=NAMES[key], color=COLORS[key], linewidth=1.4)
    bnav = benchmark.nav.to_numpy()
    ax.plot(benchmark.date, bnav / np.maximum.accumulate(np.r_[1, bnav])[1:] - 1,
            color=COLORS["benchmark"], linestyle="--", label=NAMES["benchmark"], linewidth=1.2)
    pct(ax); ax.set(title="回撤路径", ylabel="距历史最高净值的跌幅")
    ax.set_xlim(daily.date.min(), daily.date.max())
    ax.legend(loc="lower left"); ax.grid(axis="y", alpha=.16)
    finish(fig, a.output, "02_drawdown", registry, "回撤路径", ["daily_portfolios.csv", "benchmark_csi300.csv"])

    annual = read(a.results, "returns_annual.csv")
    years = sorted(annual.year.astype(int).unique())
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    x = np.arange(len(years)); width = .25
    for k, key in enumerate(["baseline", "research"]):
        vals = annual[annual.strategy == key].set_index("year").reindex(years)
        ax.bar(x + (k - 1) * width, vals.return_net, width, label=NAMES[key], color=COLORS[key])
    vals = annual[annual.strategy == "research"].set_index("year").reindex(years)
    ax.bar(x + width, vals.benchmark_return, width, label=NAMES["benchmark"], color=COLORS["benchmark"])
    ax.set_xticks(x, years); pct(ax); ax.axhline(0, color="#888888", linewidth=.7)
    ax.set(title="逐年收益对比", ylabel="年度收益")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncols=3, fontsize=9)
    ax.grid(axis="y", alpha=.12)
    finish(fig, a.output, "03_annual_returns", registry, "逐年收益对比", ["returns_annual.csv"])

    ic = read(a.results, "factor_ic_summary.csv")
    factors = list(FACTORS)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for k, (period, label, color) in enumerate([
        ("front_2018_2020", "2018–2020前段", "#54758d"),
        ("historical_holdout_2021_2025", "2021–2025历史留出段", "#bd582e")]):
        vals = ic[ic.period == period].set_index("factor").reindex(factors)
        ax.bar(np.arange(4) + (k - .5) * .32, vals.mean_rank_ic, .32, label=label, color=color)
    ax.set_xticks(np.arange(4), [FACTORS[k] for k in factors]); ax.axhline(0, color="#777777", linewidth=.8)
    ax.set(title="因子 Rank IC 的前后段比较", ylabel="月度 Rank IC 均值")
    ax.legend(); ax.grid(axis="y", alpha=.12)
    finish(fig, a.output, "04_rank_ic", registry, "因子有效性比较", ["factor_ic_summary.csv"],
           "次交易日受执行代理限制的初始买入按现金收益计；期末不完整持有期不进入汇总。")

    groups = read(a.results, "factor_groups_monthly.csv")
    groups = groups[~groups.partial_final_period.astype(bool)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), sharey=True)
    for ax, factor in zip(axes.flat, factors):
        values = groups[groups.factor == factor].groupby("group").return_gross.mean().reindex(range(1, 6))
        ax.bar(range(1, 6), values, color=["#aebdc7"] * 4 + ["#bd582e"])
        ax.set(title=FACTORS[factor], xlabel="因子从低到高的五组", xticks=range(1, 6)); pct(ax, decimals=1)
        ax.axhline(0, color="#777777", linewidth=.7); ax.grid(axis="y", alpha=.12)
    finish(fig, a.output, "05_factor_groups", registry, "五分组收益", ["factor_groups_monthly.csv"],
           "平均每期毛收益，未扣组别交易成本；固定经济方向，不根据组收益倒转买卖方向。")

    reb = read(a.results, "rebalances.csv"); reb["execution_date"] = pd.to_datetime(reb.execution_date)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for k, key in enumerate(["baseline", "research"]):
        values = reb[reb.strategy == key][factors].mean()
        ax.bar(np.arange(4) + (k - .5)*.32, values, .32, color=COLORS[key], label=NAMES[key])
    ax.set_xticks(np.arange(4), [FACTORS[k] for k in factors]); ax.axhline(0, color="#777777", linewidth=.7)
    ax.set(title="组合平均风格暴露", ylabel="持仓加权截面标准分")
    ax.legend(); ax.grid(axis="y", alpha=.12)
    finish(fig, a.output, "06_style_exposures", registry, "组合风格暴露", ["rebalances.csv"],
           "执行后实际权重加权；缺少有效因子值的旧仓由未建模仓位单独记录。")

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6), sharex=True)
    for key in ["baseline", "research", "research_no_buffer"]:
        d = reb[reb.strategy == key].sort_values("execution_date")
        label = NAMES.get(key, "研究模型无缓冲")
        color = COLORS.get(key, "#82916e")
        axes[0].plot(d.execution_date, d.one_way_turnover.rolling(3, min_periods=1).mean(), label=label, color=color)
    axes[0].set(title="调仓缓冲与实际换手", ylabel="单边换手率，3期均值"); pct(axes[0])
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncols=3, fontsize=9)
    for key in ["baseline", "research"]:
        d = daily[daily.strategy == key]
        axes[1].plot(d.date, d.transaction_cost_fraction_previous_nav.cumsum(), color=COLORS[key], label=NAMES[key])
    axes[1].set(ylabel="逐日费用率累计和"); pct(axes[1], decimals=1)
    axes[1].set_xlim(daily.date.min(), daily.date.max())
    for ax in axes: ax.grid(axis="y", alpha=.12)
    finish(fig, a.output, "07_turnover_cost", registry, "换手与费用消耗", ["rebalances.csv", "daily_portfolios.csv"],
           "费用率累计和用于展示费用强度，不等于复利净值差。初始建仓单边换手率为约50%。")

    attr = read(a.results, "return_attribution.csv")
    attr = attr[~attr.partial_final_period.astype(bool)]
    columns = ["intercept_contribution"] + [f + "_contribution" for f in factors] + ["residual_contribution"]
    labels = ["截面共同项"] + list(FACTORS.values()) + ["残差与执行差异"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for k, key in enumerate(["baseline", "research"]):
        values = attr[attr.strategy == key][columns].mean()
        ax.bar(np.arange(6) + (k-.5)*.34, values, .34, color=COLORS[key], label=NAMES[key])
    ax.set_xticks(range(6), labels, rotation=15, ha="right"); pct(ax, decimals=1)
    ax.set(title="毛收益的平均每期归因", ylabel="平均持有期贡献"); ax.axhline(0, color="#777777", linewidth=.7)
    ax.legend(); ax.grid(axis="y", alpha=.12)
    finish(fig, a.output, "08_attribution", registry, "收益来源分解", ["return_attribution.csv"],
           "截距代表横截面共同收益，不是风险调整alpha。残差吸收未建模风险和执行差异；各期闭合不等于模型解释力强。")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    bps = [5, 15, 30, 50]
    for key in ["baseline", "research"]:
        rows = [full.loc[key if b == 15 else f"{key}_cost{b}bp"] for b in bps]
        axes[0].plot(bps, [r.annualized_return for r in rows], marker="o", color=COLORS[key], label=NAMES[key])
        axes[1].plot(bps, [r.sharpe_rf0 for r in rows], marker="o", color=COLORS[key])
    axes[0].set(title="年化收益", ylabel="年化收益"); pct(axes[0], decimals=1)
    axes[1].set(title="夏普比率", ylabel="无风险利率假设为0")
    for ax in axes: ax.set_xlabel("每边综合成本，bp"); ax.set_xticks(bps); ax.grid(alpha=.15)
    axes[0].legend()
    finish(fig, a.output, "09_cost_sensitivity", registry, "成本情景检验", ["performance_metrics.csv"],
           "固定综合费用情景，不代表中国市场历年的真实税费表。")

    selected = {"baseline": "基准", "research": "研究主方案", "research_no_buffer": "无缓冲",
                "research_no_momentum": "去动量", "research_no_lowvol": "去低波动", "research_n50": "50只", "research_n150": "150只"}
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for key, label in selected.items():
        row = full.loc[key]; color = COLORS.get(key, "#8b9b9a")
        ax.scatter(row.annualized_volatility, row.annualized_return, color=color, s=85 if key == "research" else 45)
        offset = (0, -24) if key == "research_n150" else (7, 6)
        ax.annotate(label, (row.annualized_volatility, row.annualized_return), xytext=offset, textcoords="offset points", fontsize=10)
    pct(ax, decimals=1); pct(ax, axis="x", decimals=1)
    ax.margins(x=.08, y=.12)
    ax.set(title="预设扩展方案的收益与波动", xlabel="年化波动率", ylabel="年化收益率")
    ax.grid(alpha=.14)
    finish(fig, a.output, "10_prespecified_variants", registry, "预设敏感性比较", ["performance_metrics.csv"],
           "这些方案在运行前预设，主结论仍以固定100只四因子方案为准。")

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    rr = reb[reb.strategy == "research"].sort_values("execution_date")
    ax.plot(rr.execution_date, rr.capacity_min_rmb / 1e8, label="当期最紧约束", color="#bd582e")
    ax.plot(rr.execution_date, rr.capacity_p10_rmb / 1e8, label="证券约束的10%分位", color="#54758d", linestyle="--")
    ax.set_yscale("log"); ax.set(title="按流动性假设估算的规模约束", ylabel="人民币亿元，对数刻度")
    ax.set_xlim(rr.execution_date.min(), rr.execution_date.max())
    ax.legend(); ax.grid(axis="y", alpha=.15)
    finish(fig, a.output, "11_capacity", registry, "容量情景估算", ["rebalances.csv"],
           "成交额按千元解释、参与率5%、3日执行；10%分位允许最弱约束被突破，不能当作全仓可执行容量。")

    fees = read(a.results, "daily_after_product_fees.csv")
    fees["date"] = pd.to_datetime(fees.date)
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    r = daily[daily.strategy == "research"]; rf = fees[fees.strategy == "research"]
    ax.plot(r.date, r.nav_net, label="交易成本后", color="#bd582e")
    ax.plot(rf.date, rf.nav_after_assumed_product_fees, label="再扣假设产品费率后", color="#54758d", linestyle="--")
    ax.set(title="费率对持有人净值的影响", ylabel="初始资产 = 1"); ax.legend(); ax.grid(axis="y", alpha=.15)
    ax.set_xlim(r.date.min(), r.date.max())
    finish(fig, a.output, "12_product_fees", registry, "产品费率情景", ["daily_portfolios.csv", "daily_after_product_fees.csv"],
           "假设管理费0.8%/年、托管费0.1%/年，按252交易日等额计提；非真实基金收费承诺。")

    universe = read(a.results, "universe_by_signal.csv")
    date_col = "signal_date"
    count_col = next(c for c in ["eligible_count", "eligible_stock_count", "eligible"] if c in universe)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(pd.to_datetime(universe[date_col]), universe[count_col], color="#54758d", linewidth=1.8)
    ax.set(title="每次信号形成时的可选股票数", ylabel="共同股票池数量"); ax.grid(axis="y", alpha=.15)
    ax.set_xlim(pd.to_datetime(universe[date_col]).min(), pd.to_datetime(universe[date_col]).max())
    finish(fig, a.output, "13_universe", registry, "股票池随时间的变化", ["universe_by_signal.csv"],
           "按信号时点信息筛选；没有用下一期收益是否存在筛选当期股票。")

    (a.output / "figure_manifest.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(registry), "directory": str(a.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
