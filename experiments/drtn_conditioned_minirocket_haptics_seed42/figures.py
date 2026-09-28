"""
Figures for DRTN-Conditioned MiniROCKET — Haptics Seed 42

Generates diagnostic plots:
1. Performance comparison (bar chart)
2. Regime usage (histogram)
3. Feature redundancy (correlation)
4. Correctness comparison (Venn-like)
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "results", "drtn_conditioned_minirocket_haptics_seed42")


def plot_performance_comparison(results, save_path=None):
    """Bar chart comparing M0-M3 test Macro-F1."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    mf1s = [results[m]["test_macro_f1"] for m in models]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    
    bars = ax.bar(models, mf1s, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, mf1 in zip(bars, mf1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mf1:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Test Macro-F1', fontsize=12)
    ax.set_title('DRTN-Conditioned MiniROCKET — Haptics Seed 42\nModel Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(mf1s) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_regime_usage(regime_stats, save_path=None):
    """Histogram of regime occupancy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    splits = ['train', 'val', 'test']
    
    for ax, split in zip(axes, splits):
        if split in regime_stats:
            usage = regime_stats[split]["usage"]
            K = len(usage)
            codes = list(range(K))
            
            ax.bar(codes, usage, color='#2196F3', edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Regime Code', fontsize=10)
            ax.set_ylabel('Occupancy Fraction', fontsize=10)
            ax.set_title(f'{split.capitalize()} (n={regime_stats[split].get("n_samples", "?")})', 
                        fontsize=11)
            ax.set_xticks(codes)
            ax.grid(axis='y', alpha=0.3)
            
            # Add entropy annotation
            entropy = regime_stats[split]["entropy"]
            perplexity = regime_stats[split]["perplexity"]
            ax.text(0.95, 0.95, f'H={entropy:.3f}\nPP={perplexity:.2f}',
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('DRTN Regime Usage — Haptics Seed 42', 
                fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_feature_redundancy(F_global, F_heterogeneity, save_path=None):
    """Correlation between global PPV and heterogeneity features."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subsample for visualization
    n_show = min(500, F_global.shape[1])
    idx = np.random.RandomState(42).choice(F_global.shape[1], n_show, replace=False)
    
    # Correlation matrix
    corr = np.corrcoef(F_global[:, idx].T, F_heterogeneity[:, idx].T)
    n = n_show
    cross_corr = corr[:n, n:]
    
    # Histogram of cross-correlations
    ax = axes[0]
    ax.hist(cross_corr.flatten(), bins=50, color='#2196F3', edgecolor='black', 
            linewidth=0.5, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Zero')
    ax.set_xlabel('Pearson Correlation', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Cross-Block Feature Correlation', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Scatter of mean correlations per kernel
    ax = axes[1]
    mean_corr_per_kernel = np.mean(cross_corr, axis=1)
    ax.hist(mean_corr_per_kernel, bins=30, color='#4CAF50', edgecolor='black',
            linewidth=0.5, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Zero')
    ax.set_xlabel('Mean Correlation per Global Kernel', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Global Kernel → Heterogeneity Block', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.suptitle('Feature Redundancy Analysis', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_correctness_comparison(complementarity, save_path=None):
    """Venn-like visualization of M0 vs M1 correctness."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n = complementarity["n_samples"]
    both_correct = complementarity["both_correct"]
    m0_only = complementarity["M0_only_correct"]
    m1_only = complementarity["M1_only_correct"]
    both_wrong = complementarity["both_wrong"]
    
    # Simple stacked bar
    categories = ['Both Correct', 'M0 Only', 'M1 Only', 'Both Wrong']
    counts = [both_correct, m0_only, m1_only, both_wrong]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    
    bottom = 0
    for cat, count, color in zip(categories, counts, colors):
        ax.barh('Samples', count, left=bottom, color=color, edgecolor='black',
                linewidth=0.5, label=f'{cat}: {count} ({100*count/n:.1f}%)')
        if count > n * 0.05:  # Only label if >5%
            ax.text(bottom + count/2, 0, str(count), ha='center', va='center',
                   fontweight='bold', fontsize=10)
        bottom += count
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_title('M0 vs M1 Correctness Comparison\n(Haptics Test Set)', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)
    ax.set_xlim(0, n)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_per_class_f1(results, n_classes=5, save_path=None):
    """Per-class F1 comparison across models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(results.keys())
    x = np.arange(n_classes)
    width = 0.2
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    
    for i, (model, color) in enumerate(zip(models, colors)):
        f1s = results[model]["class_f1s"]
        ax.bar(x + i * width, f1s, width, label=model, color=color, 
               edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Comparison — Haptics Seed 42', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'Class {i}' for i in range(n_classes)])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def generate_all_figures(report_path=None):
    """Generate all diagnostic figures."""
    print("\n[FIGURES] Generating diagnostic plots...")
    
    # Load report
    if report_path is None:
        report_path = os.path.join(OUT_DIR, "report.json")
    
    with open(report_path) as f:
        report = json.load(f)
    
    results = report["results"]
    complementarity = report["complementarity"]
    regime_stats = report["regime_statistics"]
    
    # Load feature statistics
    with open(os.path.join(OUT_DIR, "diagnostics", "feature_statistics.json")) as f:
        feat_stats = json.load(f)
    
    # Generate figures
    fig_dir = os.path.join(OUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    plot_performance_comparison(results, 
                                os.path.join(fig_dir, "performance_comparison.png"))
    
    plot_regime_usage(regime_stats["test"],
                     os.path.join(fig_dir, "regime_usage.png"))
    
    plot_correctness_comparison(complementarity,
                               os.path.join(fig_dir, "correctness_comparison.png"))
    
    # Feature redundancy (need actual features - placeholder for now)
    # This would require loading the actual feature matrices
    
    print("  All figures generated.")


if __name__ == "__main__":
    generate_all_figures()
