# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

import os
from typing import Iterable

import pandas as pd


DEFAULT_METRICS = (
    "r_deg",
    "t_cm",
    "ADD(S)-0.1d",
    "MSSD",
    "MSPD",
    "VSD",
    "AR",
)


def compute_metric_means(df: pd.DataFrame, metrics: Iterable[str] = DEFAULT_METRICS) -> dict:
    means = {}
    for metric in metrics:
        if metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce")
        if series.notna().any():
            means[metric] = float(series.mean())
    return means


def build_runtime_summary(exp_tag: str, df: pd.DataFrame, metrics: Iterable[str] = DEFAULT_METRICS) -> str:
    means = compute_metric_means(df, metrics)
    num_samples = len(df)

    lines = [
        f"exp_tag: {exp_tag}",
        f"num_samples: {num_samples}",
        "",
        "mean_metrics:",
    ]

    for metric in metrics:
        if metric not in means:
            continue
        value = means[metric]
        if metric in {"ADD(S)-0.1d", "MSSD", "MSPD", "VSD", "AR"}:
            lines.append(f"  {metric}: {value:.6f} ({value * 100:.2f}%)")
        else:
            lines.append(f"  {metric}: {value:.6f}")

    if {"AR", "VSD", "MSSD", "MSPD", "ADD(S)-0.1d"}.issubset(means):
        latex_line = (
            f"{exp_tag} & "
            f"{means['AR'] * 100:.1f} & "
            f"{means['VSD'] * 100:.1f} & "
            f"{means['MSSD'] * 100:.1f} & "
            f"{means['MSPD'] * 100:.1f} & "
            f"{means['ADD(S)-0.1d'] * 100:.1f} & - \\\\"
        )
        lines.extend(["", "latex:", f"  {latex_line}"])

    return "\n".join(lines) + "\n"


def write_runtime_summary(output_path: str, exp_tag: str, df: pd.DataFrame, metrics: Iterable[str] = DEFAULT_METRICS) -> tuple[str, str, str]:
    summary_text = build_runtime_summary(exp_tag, df, metrics)
    output_base, _ = os.path.splitext(output_path)
    txt_path = output_base + "_summary.txt"
    log_path = output_base + "_summary.log"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return summary_text, txt_path, log_path
