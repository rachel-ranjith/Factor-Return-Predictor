"""Backtest visualization and reporting."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from backtest.walkforward import WalkForwardResult


def plot_cumulative_ic(result: WalkForwardResult, save_path: Path = None) -> None:
    """Plot cumulative IC over time."""
    metrics = result.metrics_by_period.copy()
    metrics["cumulative_ic"] = metrics["ic"].cumsum()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Period IC
    ax1 = axes[0]
    colors = ["green" if x > 0 else "red" for x in metrics["ic"]]
    ax1.bar(metrics["period"], metrics["ic"], color=colors, alpha=0.7)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_xlabel("Period")
    ax1.set_ylabel("IC")
    ax1.set_title("Information Coefficient by Period")
    
    # Cumulative IC
    ax2 = axes[1]
    ax2.plot(metrics["period"], metrics["cumulative_ic"], marker="o", markersize=3)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Cumulative IC")
    ax2.set_title("Cumulative IC Over Time")
    ax2.fill_between(
        metrics["period"],
        metrics["cumulative_ic"],
        alpha=0.3,
        where=(metrics["cumulative_ic"] > 0),
        color="green",
    )
    ax2.fill_between(
        metrics["period"],
        metrics["cumulative_ic"],
        alpha=0.3,
        where=(metrics["cumulative_ic"] < 0),
        color="red",
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_coefficient_stability(result: WalkForwardResult, save_path: Path = None) -> None:
    """Plot how coefficients change over time."""
    coef_df = result.coefficients_by_period.drop(columns=["period"])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for col in coef_df.columns:
        ax.plot(coef_df.index, coef_df[col], label=col, alpha=0.7)
    
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Period")
    ax.set_ylabel("Coefficient Value (standardized)")
    ax.set_title("Feature Coefficients Over Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_predictions_vs_actuals(result: WalkForwardResult, save_path: Path = None) -> None:
    """Scatter plot of predictions vs actual returns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter
    ax1 = axes[0]
    ax1.scatter(result.predictions, result.actuals, alpha=0.3, s=10)
    
    # Add regression line
    z = np.polyfit(result.predictions, result.actuals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(result.predictions.min(), result.predictions.max(), 100)
    ax1.plot(x_line, p(x_line), "r--", label=f"Slope: {z[0]:.3f}")
    
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_xlabel("Predicted Return")
    ax1.set_ylabel("Actual Return")
    ax1.set_title("Predictions vs Actuals")
    ax1.legend()
    
    # Distribution of predictions
    ax2 = axes[1]
    ax2.hist(result.predictions, bins=50, alpha=0.7, label="Predictions", density=True)
    ax2.hist(result.actuals, bins=50, alpha=0.7, label="Actuals", density=True)
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of Predictions vs Actuals")
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_hit_rate_over_time(result: WalkForwardResult, save_path: Path = None) -> None:
    """Plot hit rate by period with rolling average."""
    metrics = result.metrics_by_period.copy()
    metrics["rolling_hit"] = metrics["hit_rate"].rolling(5, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.bar(metrics["period"], metrics["hit_rate"], alpha=0.5, label="Period Hit Rate")
    ax.plot(metrics["period"], metrics["rolling_hit"], color="red", linewidth=2, label="5-Period Rolling Avg")
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1, label="Random (50%)")
    
    ax.set_xlabel("Period")
    ax.set_ylabel("Hit Rate")
    ax.set_title("Directional Accuracy Over Time")
    ax.set_ylim(0.2, 0.8)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def generate_report(
    result: WalkForwardResult,
    model_name: str,
    output_dir: Path = Path("analysis/plots"),
) -> None:
    """Generate full visual report for a model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    
    plot_cumulative_ic(result, output_dir / f"{prefix}_cumulative_ic.png")
    plot_coefficient_stability(result, output_dir / f"{prefix}_coefficients.png")
    plot_predictions_vs_actuals(result, output_dir / f"{prefix}_predictions.png")
    plot_hit_rate_over_time(result, output_dir / f"{prefix}_hit_rate.png")
    
    # Summary stats
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"Overall IC:       {result.overall_metrics['ic']:.4f}")
    print(f"Overall Hit Rate: {result.overall_metrics['hit_rate']:.2%}")
    print(f"Overall R²:       {result.overall_metrics['r_squared']:.4f}")
    print(f"Total Samples:    {result.overall_metrics['n_samples']}")
    print(f"\nPeriod IC Stats:")
    print(f"  Mean:   {result.metrics_by_period['ic'].mean():.4f}")
    print(f"  Std:    {result.metrics_by_period['ic'].std():.4f}")
    print(f"  Min:    {result.metrics_by_period['ic'].min():.4f}")
    print(f"  Max:    {result.metrics_by_period['ic'].max():.4f}")
    print(f"  % > 0:  {(result.metrics_by_period['ic'] > 0).mean():.1%}")