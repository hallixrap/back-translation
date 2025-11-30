"""
Back Translation Project - Results Aggregation & Matrix Generation
Stanford Clinical Translation Evaluation Framework

This module aggregates all metrics and generates:
- Summary matrices (Model × Language)
- Detailed reports
- Visualizations
- Export to various formats (Excel, CSV, HTML)
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics

from config import (
    METRICS_DIR, REPORTS_DIR, OUTPUT_DIR,
    LANGUAGES, ACTIVE_LANGUAGES, ACTIVE_MODELS,
    METRIC_THRESHOLDS, SUITABILITY_THRESHOLDS,
    DOCUMENT_CATEGORIES, logger
)

# Optional imports for visualization
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed. Some features will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("matplotlib/seaborn not installed. Plotting disabled.")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_metrics(filename: str = "all_metrics.json") -> list[dict]:
    """Load metrics from JSON file."""
    filepath = METRICS_DIR / filename

    if not filepath.exists():
        logger.error(f"Metrics file not found: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} metric records")
    return data


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_by_model_language(metrics: list[dict]) -> dict:
    """
    Aggregate metrics by model and language.

    Returns nested dict: {model: {language: {metric: [values]}}}
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for m in metrics:
        model = m['model']
        lang = m['target_language']

        # Collect all numeric metrics
        for key, value in m.items():
            if isinstance(value, (int, float)) and value is not None:
                aggregated[model][lang][key].append(value)

    return aggregated


def calculate_summary_stats(values: list) -> dict:
    """Calculate summary statistics for a list of values."""
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
        "min": min(values),
        "max": max(values),
        "n": len(values)
    }


def generate_summary_matrix(metrics: list[dict], metric_name: str = "composite_score") -> dict:
    """
    Generate a summary matrix for a specific metric.

    Returns:
        {
            "matrix": {model: {language: mean_score}},
            "models": [model_list],
            "languages": [language_list],
            "metric": metric_name
        }
    """
    aggregated = aggregate_by_model_language(metrics)

    models = sorted(set(m['model'] for m in metrics))
    languages = sorted(set(m['target_language'] for m in metrics))

    matrix = {}
    for model in models:
        matrix[model] = {}
        for lang in languages:
            values = aggregated[model][lang].get(metric_name, [])
            if values:
                matrix[model][lang] = statistics.mean(values)
            else:
                matrix[model][lang] = None

    return {
        "matrix": matrix,
        "models": models,
        "languages": languages,
        "metric": metric_name
    }


def generate_full_report(metrics: list[dict]) -> dict:
    """
    Generate a comprehensive report with all aggregations.
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_evaluations": len(metrics),
        "models_evaluated": list(set(m['model'] for m in metrics)),
        "languages_evaluated": list(set(m['target_language'] for m in metrics)),
        "documents_evaluated": list(set(m['doc_id'] for m in metrics)),
    }

    # Summary matrices for each key metric
    key_metrics = ["composite_score", "bertscore_f1", "bleu", "chrf", "labse_similarity"]
    report["summary_matrices"] = {}

    for metric in key_metrics:
        report["summary_matrices"][metric] = generate_summary_matrix(metrics, metric)

    # Suitability summary
    suitability_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for m in metrics:
        model = m['model']
        lang = m['target_language']
        rating = m.get('suitability_rating', 'unknown')
        suitability_counts[model][lang][rating] += 1

    report["suitability_distribution"] = dict(suitability_counts)

    # Best/worst performers
    report["rankings"] = generate_rankings(metrics)

    # Category breakdown (if documents have categories)
    report["by_category"] = aggregate_by_category(metrics)

    return report


def generate_rankings(metrics: list[dict]) -> dict:
    """Generate rankings for models and languages."""

    # Rank models (by average composite score)
    model_scores = defaultdict(list)
    for m in metrics:
        if m.get('composite_score') is not None:
            model_scores[m['model']].append(m['composite_score'])

    model_ranking = [
        {"model": model, "avg_score": statistics.mean(scores), "n": len(scores)}
        for model, scores in model_scores.items()
    ]
    model_ranking.sort(key=lambda x: x['avg_score'], reverse=True)

    # Rank languages (by average composite score across all models)
    lang_scores = defaultdict(list)
    for m in metrics:
        if m.get('composite_score') is not None:
            lang_scores[m['target_language']].append(m['composite_score'])

    lang_ranking = [
        {
            "language": lang,
            "avg_score": statistics.mean(scores),
            "n": len(scores),
            "language_name": LANGUAGES.get(lang, {}).get('name', lang)
        }
        for lang, scores in lang_scores.items()
    ]
    lang_ranking.sort(key=lambda x: x['avg_score'], reverse=True)

    # Best model-language combinations
    combo_scores = defaultdict(list)
    for m in metrics:
        if m.get('composite_score') is not None:
            key = f"{m['model']}|{m['target_language']}"
            combo_scores[key].append(m['composite_score'])

    combo_ranking = [
        {
            "combination": key,
            "model": key.split('|')[0],
            "language": key.split('|')[1],
            "avg_score": statistics.mean(scores),
            "n": len(scores)
        }
        for key, scores in combo_scores.items()
    ]
    combo_ranking.sort(key=lambda x: x['avg_score'], reverse=True)

    return {
        "by_model": model_ranking,
        "by_language": lang_ranking,
        "by_combination": combo_ranking[:10],  # Top 10
        "worst_combinations": combo_ranking[-10:]  # Bottom 10
    }


def aggregate_by_category(metrics: list[dict]) -> dict:
    """Aggregate metrics by document category."""
    # This requires doc_id to category mapping
    # For now, extract category from doc_id prefix if available

    category_scores = defaultdict(list)

    for m in metrics:
        doc_id = m.get('doc_id', '')
        # Try to extract category from doc_id (e.g., "CARD_001" -> "cardiology")
        prefix_map = {
            "CARD": "cardiology",
            "DIAB": "diabetes",
            "RESP": "respiratory",
            "MED": "medications",
            "PREV": "preventive_care",
            "EMER": "emergency_care",
            "SURG": "surgical",
            "MH": "mental_health"
        }

        category = "unknown"
        for prefix, cat in prefix_map.items():
            if doc_id.startswith(prefix):
                category = cat
                break

        if m.get('composite_score') is not None:
            category_scores[category].append(m['composite_score'])

    return {
        cat: calculate_summary_stats(scores)
        for cat, scores in category_scores.items()
    }


# =============================================================================
# MATRIX DISPLAY & EXPORT
# =============================================================================

def print_matrix(matrix_data: dict, show_suitability: bool = True):
    """Print a formatted matrix to console."""
    matrix = matrix_data['matrix']
    models = matrix_data['models']
    languages = matrix_data['languages']
    metric = matrix_data['metric']

    print(f"\n{'='*80}")
    print(f"SUMMARY MATRIX: {metric.upper()}")
    print(f"{'='*80}")

    # Header row
    lang_names = [LANGUAGES.get(l, {}).get('name', l)[:10] for l in languages]
    header = "Model".ljust(20) + "".join(l.center(12) for l in lang_names)
    print(header)
    print("-" * len(header))

    # Data rows
    for model in models:
        row = model[:18].ljust(20)
        for lang in languages:
            value = matrix[model].get(lang)
            if value is not None:
                # Add suitability indicator
                if show_suitability and metric == "composite_score":
                    if value >= 0.80:
                        indicator = "✓"
                    elif value >= 0.60:
                        indicator = "⚠"
                    else:
                        indicator = "✗"
                    cell = f"{value:.2f}{indicator}"
                else:
                    cell = f"{value:.2f}"
            else:
                cell = "N/A"
            row += cell.center(12)
        print(row)

    print("-" * len(header))

    if show_suitability and metric == "composite_score":
        print("\nLegend: ✓ Suitable (≥0.80) | ⚠ Caution (0.60-0.79) | ✗ Not Recommended (<0.60)")


def export_to_csv(report: dict, filename: str = "translation_report.csv"):
    """Export summary matrix to CSV."""
    if not HAS_PANDAS:
        logger.error("pandas required for CSV export")
        return None

    # Get the composite score matrix
    matrix_data = report['summary_matrices']['composite_score']

    # Convert to DataFrame
    df = pd.DataFrame(matrix_data['matrix']).T

    # Rename columns to language names
    df.columns = [LANGUAGES.get(c, {}).get('name', c) for c in df.columns]

    filepath = REPORTS_DIR / filename
    df.to_csv(filepath)
    logger.info(f"Exported CSV to {filepath}")
    return filepath


def export_to_excel(report: dict, filename: str = "translation_report.xlsx"):
    """Export full report to Excel with multiple sheets."""
    if not HAS_PANDAS:
        logger.error("pandas required for Excel export")
        return None

    filepath = REPORTS_DIR / filename

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Summary sheet
        summary_df = pd.DataFrame({
            'Metric': ['Total Evaluations', 'Models', 'Languages', 'Documents'],
            'Value': [
                report['total_evaluations'],
                len(report['models_evaluated']),
                len(report['languages_evaluated']),
                len(report['documents_evaluated'])
            ]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Matrix sheets for each metric
        for metric_name, matrix_data in report['summary_matrices'].items():
            df = pd.DataFrame(matrix_data['matrix']).T
            df.columns = [LANGUAGES.get(c, {}).get('name', c) for c in df.columns]
            df.to_excel(writer, sheet_name=metric_name[:31])  # Excel sheet name limit

        # Rankings sheet
        model_ranking_df = pd.DataFrame(report['rankings']['by_model'])
        model_ranking_df.to_excel(writer, sheet_name='Model Rankings', index=False)

        lang_ranking_df = pd.DataFrame(report['rankings']['by_language'])
        lang_ranking_df.to_excel(writer, sheet_name='Language Rankings', index=False)

        combo_df = pd.DataFrame(report['rankings']['by_combination'])
        combo_df.to_excel(writer, sheet_name='Top Combinations', index=False)

    logger.info(f"Exported Excel to {filepath}")
    return filepath


def export_to_html(report: dict, filename: str = "translation_report.html"):
    """Export report to styled HTML."""
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<title>Back Translation Evaluation Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 40px; }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #34495e; border-bottom: 2px solid #3498db; }",
        "table { border-collapse: collapse; margin: 20px 0; }",
        "th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }",
        "th { background-color: #3498db; color: white; }",
        ".suitable { background-color: #2ecc71; color: white; }",
        ".caution { background-color: #f39c12; color: white; }",
        ".not-recommended { background-color: #e74c3c; color: white; }",
        ".summary-box { background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }",
        "</style>",
        "</head><body>",
        f"<h1>🔬 Back Translation Evaluation Report</h1>",
        f"<p>Generated: {report['generated_at']}</p>",
    ]

    # Summary box
    html_parts.append("<div class='summary-box'>")
    html_parts.append(f"<strong>Total Evaluations:</strong> {report['total_evaluations']}<br>")
    html_parts.append(f"<strong>Models:</strong> {', '.join(report['models_evaluated'])}<br>")
    html_parts.append(f"<strong>Languages:</strong> {len(report['languages_evaluated'])}<br>")
    html_parts.append(f"<strong>Documents:</strong> {len(report['documents_evaluated'])}")
    html_parts.append("</div>")

    # Composite Score Matrix
    html_parts.append("<h2>Composite Score Matrix</h2>")
    matrix_data = report['summary_matrices']['composite_score']

    html_parts.append("<table>")
    # Header
    html_parts.append("<tr><th>Model</th>")
    for lang in matrix_data['languages']:
        lang_name = LANGUAGES.get(lang, {}).get('name', lang)
        html_parts.append(f"<th>{lang_name}</th>")
    html_parts.append("</tr>")

    # Data rows
    for model in matrix_data['models']:
        html_parts.append(f"<tr><td><strong>{model}</strong></td>")
        for lang in matrix_data['languages']:
            value = matrix_data['matrix'][model].get(lang)
            if value is not None:
                if value >= 0.80:
                    css_class = "suitable"
                elif value >= 0.60:
                    css_class = "caution"
                else:
                    css_class = "not-recommended"
                html_parts.append(f"<td class='{css_class}'>{value:.3f}</td>")
            else:
                html_parts.append("<td>N/A</td>")
        html_parts.append("</tr>")

    html_parts.append("</table>")

    # Legend
    html_parts.append("<p><span class='suitable' style='padding:3px 8px;'>✓ Suitable (≥0.80)</span> ")
    html_parts.append("<span class='caution' style='padding:3px 8px;'>⚠ Caution (0.60-0.79)</span> ")
    html_parts.append("<span class='not-recommended' style='padding:3px 8px;'>✗ Not Recommended (<0.60)</span></p>")

    # Rankings
    html_parts.append("<h2>Model Rankings</h2>")
    html_parts.append("<table><tr><th>Rank</th><th>Model</th><th>Avg Score</th><th>N</th></tr>")
    for i, r in enumerate(report['rankings']['by_model'], 1):
        html_parts.append(f"<tr><td>{i}</td><td>{r['model']}</td><td>{r['avg_score']:.3f}</td><td>{r['n']}</td></tr>")
    html_parts.append("</table>")

    html_parts.append("<h2>Language Rankings</h2>")
    html_parts.append("<table><tr><th>Rank</th><th>Language</th><th>Avg Score</th><th>N</th></tr>")
    for i, r in enumerate(report['rankings']['by_language'], 1):
        html_parts.append(f"<tr><td>{i}</td><td>{r['language_name']}</td><td>{r['avg_score']:.3f}</td><td>{r['n']}</td></tr>")
    html_parts.append("</table>")

    # Close HTML
    html_parts.append("</body></html>")

    # Write file
    filepath = REPORTS_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_parts))

    logger.info(f"Exported HTML to {filepath}")
    return filepath


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_heatmap(report: dict, metric: str = "composite_score", save_path: str = None):
    """Generate a heatmap visualization of the matrix."""
    if not HAS_PLOTTING or not HAS_PANDAS:
        logger.warning("matplotlib and pandas required for plotting")
        return

    matrix_data = report['summary_matrices'][metric]
    df = pd.DataFrame(matrix_data['matrix']).T
    df.columns = [LANGUAGES.get(c, {}).get('name', c) for c in df.columns]

    plt.figure(figsize=(14, 8))

    # Create heatmap
    sns.heatmap(
        df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0.7,
        vmin=0.3,
        vmax=1.0,
        linewidths=0.5
    )

    plt.title(f'Translation Quality Matrix: {metric.replace("_", " ").title()}', fontsize=14)
    plt.xlabel('Target Language', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()

    if save_path:
        filepath = REPORTS_DIR / save_path
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved heatmap to {filepath}")
    else:
        plt.show()

    plt.close()


def plot_model_comparison(report: dict, save_path: str = None):
    """Generate a bar chart comparing model performance."""
    if not HAS_PLOTTING:
        logger.warning("matplotlib required for plotting")
        return

    rankings = report['rankings']['by_model']

    models = [r['model'] for r in rankings]
    scores = [r['avg_score'] for r in rankings]

    colors = ['#2ecc71' if s >= 0.8 else '#f39c12' if s >= 0.6 else '#e74c3c' for s in scores]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(models, scores, color=colors)
    plt.xlabel('Average Composite Score')
    plt.title('Model Performance Comparison')
    plt.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Suitable threshold')
    plt.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='Caution threshold')
    plt.legend()
    plt.tight_layout()

    if save_path:
        filepath = REPORTS_DIR / save_path
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved model comparison to {filepath}")
    else:
        plt.show()

    plt.close()


def plot_language_comparison(report: dict, save_path: str = None):
    """Generate a bar chart comparing language performance."""
    if not HAS_PLOTTING:
        logger.warning("matplotlib required for plotting")
        return

    rankings = report['rankings']['by_language']

    languages = [r['language_name'] for r in rankings]
    scores = [r['avg_score'] for r in rankings]

    colors = ['#2ecc71' if s >= 0.8 else '#f39c12' if s >= 0.6 else '#e74c3c' for s in scores]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(languages, scores, color=colors)
    plt.ylabel('Average Composite Score')
    plt.title('Translation Quality by Target Language')
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Suitable threshold')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Caution threshold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    if save_path:
        filepath = REPORTS_DIR / save_path
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved language comparison to {filepath}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# KEVIN'S SCORECARD FORMAT
# =============================================================================

def generate_kevin_scorecard(report: dict) -> str:
    """
    Generate the scorecard format Kevin wants to see.

    This is the "deliverable" format for the Stanford team.
    """
    lines = []
    lines.append("="*70)
    lines.append("LLM TRANSLATION SUITABILITY MATRIX")
    lines.append("Stanford Clinical Translation Evaluation Framework")
    lines.append("="*70)
    lines.append(f"\nGenerated: {report['generated_at']}")
    lines.append(f"Documents Evaluated: {len(report['documents_evaluated'])}")
    lines.append(f"Total Evaluations: {report['total_evaluations']}")
    lines.append("")

    # Main matrix
    matrix_data = report['summary_matrices']['composite_score']
    models = matrix_data['models']
    languages = matrix_data['languages']

    # Build header
    lang_headers = [LANGUAGES.get(l, {}).get('name', l)[:8] for l in languages]
    header = "".ljust(18) + " | ".join(h.center(8) for h in lang_headers)
    lines.append(header)
    lines.append("-" * len(header))

    # Build rows
    for model in models:
        row = model[:16].ljust(18)
        cells = []
        for lang in languages:
            value = matrix_data['matrix'][model].get(lang)
            if value is not None:
                if value >= 0.80:
                    cell = f"✅{value:.2f}"
                elif value >= 0.60:
                    cell = f"⚠️{value:.2f}"
                else:
                    cell = f"❌{value:.2f}"
            else:
                cell = "  N/A  "
            cells.append(cell.center(8))
        row += " | ".join(cells)
        lines.append(row)

    lines.append("-" * len(header))
    lines.append("")
    lines.append("Legend: ✅ Suitable (≥0.80) | ⚠️ Caution (0.60-0.79) | ❌ Not Recommended (<0.60)")
    lines.append("")

    # Key findings
    lines.append("="*70)
    lines.append("KEY FINDINGS")
    lines.append("="*70)

    # Best model
    best_model = report['rankings']['by_model'][0]
    lines.append(f"\n🥇 Top Performing Model: {best_model['model']} (avg: {best_model['avg_score']:.3f})")

    # Worst language
    worst_lang = report['rankings']['by_language'][-1]
    lines.append(f"\n⚠️ Most Challenging Language: {worst_lang['language_name']} (avg: {worst_lang['avg_score']:.3f})")

    # Suitability summary
    total_suitable = sum(
        1 for m, langs in report['summary_matrices']['composite_score']['matrix'].items()
        for l, v in langs.items() if v and v >= 0.80
    )
    total_combos = len(models) * len(languages)
    lines.append(f"\n📊 Combinations rated 'Suitable': {total_suitable}/{total_combos} ({100*total_suitable/total_combos:.1f}%)")

    lines.append("")
    lines.append("="*70)

    return "\n".join(lines)


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def generate_all_reports(metrics_file: str = "all_metrics.json"):
    """Generate all reports and exports."""
    metrics = load_all_metrics(metrics_file)

    if not metrics:
        logger.error("No metrics to report on")
        return

    # Generate report
    report = generate_full_report(metrics)

    # Save JSON report
    report_path = REPORTS_DIR / "full_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved JSON report to {report_path}")

    # Print matrix
    print_matrix(report['summary_matrices']['composite_score'])

    # Generate Kevin's scorecard
    scorecard = generate_kevin_scorecard(report)
    print("\n" + scorecard)

    scorecard_path = REPORTS_DIR / "kevin_scorecard.txt"
    with open(scorecard_path, 'w', encoding='utf-8') as f:
        f.write(scorecard)
    logger.info(f"Saved scorecard to {scorecard_path}")

    # Export to various formats
    export_to_csv(report)
    export_to_excel(report)
    export_to_html(report)

    # Generate plots if available
    if HAS_PLOTTING:
        plot_heatmap(report, save_path="heatmap_composite.png")
        plot_model_comparison(report, save_path="model_comparison.png")
        plot_language_comparison(report, save_path="language_comparison.png")

    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]
    else:
        metrics_file = "all_metrics.json"

    generate_all_reports(metrics_file)
