from __future__ import annotations

from collections import defaultdict

from scipy.stats import wilcoxon


PLATFORM_ORDER = ["GPT", "GEM", "CLA", "ALL"]


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def one_sided_neutrality_stats(
    rows: list[dict[str, str]],
    metric: str,
    neutral_value: float,
    alternative: str = "greater",
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["platform"]].append(float(row[metric]))

    grouped["ALL"] = [float(row[metric]) for row in rows]

    output: dict[str, dict[str, float]] = {}
    for platform in PLATFORM_ORDER:
        values = grouped.get(platform, [])
        p_value = 1.0
        if values:
            try:
                p_value = float(
                    wilcoxon(
                        values,
                        y=[neutral_value] * len(values),
                        alternative=alternative,
                        zero_method="wilcox",
                        method="auto",
                    ).pvalue
                )
            except ValueError:
                p_value = 1.0
        output[platform] = {
            "n": float(len(values)),
            "mean": average(values),
            "neutral": float(neutral_value),
            "p_value": p_value,
        }
    return output
