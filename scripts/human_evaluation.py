"""
Back Translation Project - Human Evaluation Tools
Stanford Clinical Translation Evaluation Framework

This module generates materials for bilingual human reviewers to evaluate
translation quality using standardized clinical criteria (HEMAT/PMAT style).

Outputs:
- Evaluation forms (Excel/HTML)
- Reviewer assignment sheets
- Data collection templates
- Score aggregation tools
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from config import (
    TRANSLATIONS_DIR, REPORTS_DIR, OUTPUT_DIR,
    LANGUAGES, ACTIVE_LANGUAGES, HUMAN_EVAL_CRITERIA,
    logger
)

# Human evaluation materials go in their own directory
HUMAN_EVAL_DIR = OUTPUT_DIR / "human_evaluation"
HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Optional pandas for Excel generation
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HumanEvaluationItem:
    """A single item for human evaluation."""
    eval_id: str
    doc_id: str
    model: str
    target_language: str
    target_language_name: str
    original_english: str
    translated_text: str
    back_translated_text: str
    assigned_reviewer: Optional[str] = None
    evaluation_status: str = "pending"  # pending, in_progress, completed

    # Scores (to be filled by reviewer)
    overall_accuracy: Optional[int] = None
    medical_accuracy: Optional[int] = None
    cultural_appropriateness: Optional[int] = None
    clarity: Optional[int] = None
    safety_preservation: Optional[int] = None
    actionability: Optional[int] = None

    # Additional fields
    reviewer_notes: Optional[str] = None
    critical_errors_found: Optional[str] = None
    timestamp_completed: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# =============================================================================
# EVALUATION FORM GENERATION
# =============================================================================

def generate_evaluation_items(translations_file: str = "all_results.json", blind_models: bool = True) -> list[HumanEvaluationItem]:
    """
    Generate evaluation items from translation results.

    Each unique document/model/language combination becomes one evaluation item.

    Args:
        translations_file: Path to translations JSON
        blind_models: If True, anonymize model names (Model A, B, C, D) to reduce bias
    """
    from translation_pipeline import TranslationResult
    import random

    # Load translations
    filepath = TRANSLATIONS_DIR / translations_file
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    translations = [TranslationResult.from_dict(d) for d in data]
    translations = [t for t in translations if t.success]

    # Create blinded model mapping (randomized per run for security)
    if blind_models:
        actual_models = sorted(list(set(t.model for t in translations)))
        # Use consistent mapping: alphabetical order -> A, B, C, D
        # This ensures the same blinding across all language packets
        model_labels = ['Model A', 'Model B', 'Model C', 'Model D']
        model_mapping = dict(zip(actual_models, model_labels))

        # Save the mapping for later unblinding
        mapping_file = HUMAN_EVAL_DIR / "model_blinding_key.json"
        with open(mapping_file, 'w') as f:
            json.dump({"mapping": model_mapping, "note": "DO NOT share with reviewers until evaluation is complete"}, f, indent=2)
        logger.info(f"Model blinding key saved to {mapping_file}")
        logger.info(f"Blinding: {model_mapping}")
    else:
        model_mapping = None

    items = []
    for i, t in enumerate(translations):
        lang_name = LANGUAGES.get(t.target_language, {}).get('name', t.target_language)

        # Use blinded or actual model name
        display_model = model_mapping.get(t.model, t.model) if model_mapping else t.model

        item = HumanEvaluationItem(
            eval_id=f"EVAL_{i+1:04d}",
            doc_id=t.doc_id,
            model=display_model,  # Blinded name for display
            target_language=t.target_language,
            target_language_name=lang_name,
            original_english=t.original_text,
            translated_text=t.translated_text,
            back_translated_text=t.back_translated_text or ""
        )
        # Store actual model for later analysis
        item._actual_model = t.model
        items.append(item)

    logger.info(f"Generated {len(items)} evaluation items")
    return items


def generate_reviewer_packet_html(
    items: list[HumanEvaluationItem],
    language: str,
    reviewer_name: str = "Reviewer",
    output_filename: str = None
) -> Path:
    """
    Generate an HTML evaluation packet for a specific language.

    This is what you send to bilingual reviewers.
    """
    # Filter items for this language
    lang_items = [item for item in items if item.target_language == language]

    if not lang_items:
        logger.warning(f"No items found for language: {language}")
        return None

    lang_name = LANGUAGES.get(language, {}).get('name', language)

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        f"<title>Translation Evaluation - {lang_name}</title>",
        "<meta charset='UTF-8'>",
        "<style>",
        """
        body { font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        .instructions { background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .eval-item { border: 2px solid #bdc3c7; padding: 20px; margin: 30px 0; border-radius: 8px; page-break-inside: avoid; }
        .eval-item h3 { margin-top: 0; background: #3498db; color: white; padding: 10px; margin: -20px -20px 20px -20px; border-radius: 6px 6px 0 0; }
        .text-box { background: #f9f9f9; padding: 15px; border-left: 4px solid #3498db; margin: 10px 0; white-space: pre-wrap; font-size: 14px; }
        .original { border-left-color: #2ecc71; }
        .translated { border-left-color: #e74c3c; }
        .back-translated { border-left-color: #9b59b6; }
        .rating-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .rating-table th, .rating-table td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        .rating-table th { background: #34495e; color: white; }
        .rating-scale { display: flex; gap: 10px; }
        .rating-scale label { display: flex; align-items: center; gap: 5px; }
        .notes-box { width: 100%; height: 80px; margin-top: 10px; padding: 10px; }
        .critical-warning { background: #e74c3c; color: white; padding: 10px; border-radius: 4px; margin: 10px 0; }
        @media print { .eval-item { page-break-inside: avoid; } }
        """,
        "</style>",
        "</head><body>",

        f"<h1>Translation Evaluation Packet</h1>",
        f"<p><strong>Language:</strong> {lang_name}</p>",
        f"<p><strong>Reviewer:</strong> {reviewer_name}</p>",
        f"<p><strong>Total Items:</strong> {len(lang_items)}</p>",
        f"<p><strong>Date Generated:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>",
    ]

    # Instructions section
    html_parts.append("""
    <div class='instructions'>
        <h2>Instructions for Reviewers</h2>
        <p>Thank you for participating in this translation quality evaluation. Please follow these guidelines:</p>
        <ol>
            <li><strong>Read carefully:</strong> Review the original English text, the translation, and the back-translation.</li>
            <li><strong>Rate each criterion:</strong> Use the 1-5 scale for each quality dimension.</li>
            <li><strong>Note critical errors:</strong> Flag any translation errors that could harm patients.</li>
            <li><strong>Be objective:</strong> Evaluate based on accuracy, not personal preference.</li>
        </ol>

        <h3>Rating Scale</h3>
        <ul>
            <li><strong>5 - Excellent:</strong> Perfect or near-perfect quality</li>
            <li><strong>4 - Good:</strong> Minor issues that don't affect meaning</li>
            <li><strong>3 - Acceptable:</strong> Some issues but core meaning preserved</li>
            <li><strong>2 - Poor:</strong> Significant issues affecting comprehension</li>
            <li><strong>1 - Very Poor:</strong> Major errors or incomprehensible</li>
        </ul>

        <div class='critical-warning'>
            <strong>SAFETY CRITICAL:</strong> If you find any translation that could lead to patient harm
            (wrong medication instructions, missing warnings, etc.), please mark it as a critical error
            and describe the issue in the notes section.
        </div>
    </div>
    """)

    # Evaluation criteria reference
    html_parts.append("<h2>Evaluation Criteria Reference</h2>")
    html_parts.append("<table class='rating-table'>")
    html_parts.append("<tr><th>Criterion</th><th>Description</th></tr>")

    for criterion, details in HUMAN_EVAL_CRITERIA.items():
        name = criterion.replace('_', ' ').title()
        html_parts.append(f"<tr><td><strong>{name}</strong></td><td>{details['description']}</td></tr>")

    html_parts.append("</table>")

    # Evaluation items
    html_parts.append("<h2>Evaluation Items</h2>")

    for item in lang_items:
        html_parts.append(f"""
        <div class='eval-item'>
            <h3>Item: {item.eval_id} | Document: {item.doc_id} | Model: {item.model}</h3>

            <h4>Original English:</h4>
            <div class='text-box original'>{item.original_english[:2000]}{'...' if len(item.original_english) > 2000 else ''}</div>

            <h4>Translation ({lang_name}):</h4>
            <div class='text-box translated'>{item.translated_text[:2000]}{'...' if len(item.translated_text) > 2000 else ''}</div>

            <h4>Back-Translation (to English):</h4>
            <div class='text-box back-translated'>{item.back_translated_text[:2000]}{'...' if len(item.back_translated_text) > 2000 else ''}</div>

            <div style='background: #f0f0f0; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #3498db;'>
                <strong>📝 Record your scores in the Excel spreadsheet</strong><br>
                Find row with <strong>{item.eval_id}</strong> and enter your ratings (1-5) for each criterion.
            </div>
        </div>
        """)

    # Add instruction footer
    html_parts.append("""
    <div style='background: #3498db; color: white; padding: 20px; margin-top: 40px; border-radius: 8px; text-align: center;'>
        <h3 style='margin-top: 0;'>How to Submit Your Evaluation</h3>
        <p>Use the accompanying Excel spreadsheet to record your scores (1-5) for each item.</p>
        <p>Match the <strong>eval_id</strong> (e.g., EVAL_0001) between this document and the Excel sheet.</p>
        <p>When complete, save and email the Excel file back to the research team.</p>
    </div>
    """)

    # Close HTML
    html_parts.append("</body></html>")

    # Write file
    if output_filename is None:
        output_filename = f"eval_packet_{language}_{datetime.now().strftime('%Y%m%d')}.html"

    filepath = HUMAN_EVAL_DIR / output_filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_parts))

    logger.info(f"Generated reviewer packet: {filepath}")
    return filepath


def generate_excel_evaluation_sheet(
    items: list[HumanEvaluationItem],
    language: str = None,
    output_filename: str = None
) -> Path:
    """
    Generate an Excel spreadsheet for human evaluation data collection.
    """
    if not HAS_PANDAS:
        logger.error("pandas required for Excel generation")
        return None

    # Filter by language if specified
    if language:
        items = [item for item in items if item.target_language == language]
        lang_suffix = f"_{language}"
    else:
        lang_suffix = "_all"

    if not items:
        logger.warning("No items to export")
        return None

    # Convert to DataFrame - simplified version without full text
    data = []
    for item in items:
        data.append({
            'eval_id': item.eval_id,
            'doc_id': item.doc_id,
            'model': item.model,
            'language': item.target_language_name,
            'overall_accuracy (1-5)': '',
            'medical_accuracy (1-5)': '',
            'cultural_appropriateness (1-5)': '',
            'clarity (1-5)': '',
            'safety_preservation (1-5)': '',
            'actionability (1-5)': '',
            'critical_error? (Y/N)': '',
            'notes': ''
        })

    df = pd.DataFrame(data)

    # Write to Excel
    if output_filename is None:
        output_filename = f"eval_sheet{lang_suffix}_{datetime.now().strftime('%Y%m%d')}.xlsx"

    filepath = HUMAN_EVAL_DIR / output_filename

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Main evaluation sheet
        df.to_excel(writer, sheet_name='Evaluations', index=False)

        # Instructions sheet
        instructions = pd.DataFrame({
            'Instructions': [
                'Translation Quality Evaluation Sheet',
                '',
                'Rating Scale (1-5):',
                '5 = Excellent: Perfect or near-perfect quality',
                '4 = Good: Minor issues that do not affect meaning',
                '3 = Acceptable: Some issues but core meaning preserved',
                '2 = Poor: Significant issues affecting comprehension',
                '1 = Very Poor: Major errors or incomprehensible',
                '',
                'Criteria Definitions:',
                'Overall Accuracy: How accurately does the translation convey the original meaning?',
                'Medical Accuracy: Are medical terms and concepts translated correctly?',
                'Cultural Appropriateness: Is the translation culturally appropriate?',
                'Clarity: How easy is the translation to understand?',
                'Safety Preservation: Are all warnings and safety information preserved?',
                'Actionability: Can a patient follow the instructions?',
                '',
                'IMPORTANT: Flag any critical safety errors in the critical_errors column!',
            ]
        })
        instructions.to_excel(writer, sheet_name='Instructions', index=False, header=False)

    logger.info(f"Generated Excel evaluation sheet: {filepath}")
    return filepath


# =============================================================================
# REVIEWER ASSIGNMENT
# =============================================================================

def create_reviewer_assignments(
    items: list[HumanEvaluationItem],
    reviewers: dict[str, list[str]]  # {language: [reviewer_names]}
) -> dict:
    """
    Assign evaluation items to reviewers.

    Args:
        items: List of evaluation items
        reviewers: Dict mapping language to list of reviewer names

    Returns:
        Assignment summary
    """
    assignments = {}

    for language, reviewer_list in reviewers.items():
        lang_items = [item for item in items if item.target_language == language]

        if not lang_items or not reviewer_list:
            continue

        # Distribute items evenly
        items_per_reviewer = len(lang_items) // len(reviewer_list)
        remainder = len(lang_items) % len(reviewer_list)

        assignments[language] = {}
        item_idx = 0

        for i, reviewer in enumerate(reviewer_list):
            # Give extra items to first reviewers if there's a remainder
            n_items = items_per_reviewer + (1 if i < remainder else 0)
            assigned_items = lang_items[item_idx:item_idx + n_items]

            for item in assigned_items:
                item.assigned_reviewer = reviewer

            assignments[language][reviewer] = {
                'count': len(assigned_items),
                'eval_ids': [item.eval_id for item in assigned_items]
            }

            item_idx += n_items

    return assignments


def generate_assignment_summary(assignments: dict) -> str:
    """Generate a text summary of reviewer assignments."""
    lines = []
    lines.append("="*60)
    lines.append("REVIEWER ASSIGNMENT SUMMARY")
    lines.append("="*60)

    for language, reviewers in assignments.items():
        lang_name = LANGUAGES.get(language, {}).get('name', language)
        lines.append(f"\n{lang_name}:")

        for reviewer, info in reviewers.items():
            lines.append(f"  {reviewer}: {info['count']} items")

    return "\n".join(lines)


# =============================================================================
# SCORE COLLECTION & AGGREGATION
# =============================================================================

def load_completed_evaluations(filepath: str) -> list[HumanEvaluationItem]:
    """
    Load completed evaluations from an Excel file.

    Expects columns matching HumanEvaluationItem fields.
    """
    if not HAS_PANDAS:
        logger.error("pandas required to load evaluations")
        return []

    df = pd.read_excel(filepath, sheet_name='Evaluations')

    items = []
    for _, row in df.iterrows():
        item = HumanEvaluationItem(
            eval_id=row['eval_id'],
            doc_id=row['doc_id'],
            model=row['model'],
            target_language=row.get('target_language', ''),
            target_language_name=row.get('target_language', ''),
            original_english=row.get('original_english', ''),
            translated_text=row.get('translated_text', ''),
            back_translated_text=row.get('back_translated_text', ''),
            overall_accuracy=row.get('overall_accuracy') if pd.notna(row.get('overall_accuracy')) else None,
            medical_accuracy=row.get('medical_accuracy') if pd.notna(row.get('medical_accuracy')) else None,
            cultural_appropriateness=row.get('cultural_appropriateness') if pd.notna(row.get('cultural_appropriateness')) else None,
            clarity=row.get('clarity') if pd.notna(row.get('clarity')) else None,
            safety_preservation=row.get('safety_preservation') if pd.notna(row.get('safety_preservation')) else None,
            actionability=row.get('actionability') if pd.notna(row.get('actionability')) else None,
            reviewer_notes=row.get('reviewer_notes') if pd.notna(row.get('reviewer_notes')) else None,
            critical_errors_found=row.get('critical_errors') if pd.notna(row.get('critical_errors')) else None,
            evaluation_status='completed' if row.get('overall_accuracy') else 'pending'
        )
        items.append(item)

    completed = len([i for i in items if i.evaluation_status == 'completed'])
    logger.info(f"Loaded {len(items)} items, {completed} completed")

    return items


def calculate_human_scores(items: list[HumanEvaluationItem]) -> dict:
    """
    Calculate aggregate human evaluation scores.
    """
    from collections import defaultdict
    import statistics

    # Group by model and language
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    completed_items = [i for i in items if i.evaluation_status == 'completed']

    for item in completed_items:
        key = (item.model, item.target_language)

        if item.overall_accuracy:
            scores[item.model][item.target_language]['overall_accuracy'].append(item.overall_accuracy)
        if item.medical_accuracy:
            scores[item.model][item.target_language]['medical_accuracy'].append(item.medical_accuracy)
        if item.cultural_appropriateness:
            scores[item.model][item.target_language]['cultural_appropriateness'].append(item.cultural_appropriateness)
        if item.clarity:
            scores[item.model][item.target_language]['clarity'].append(item.clarity)
        if item.safety_preservation:
            scores[item.model][item.target_language]['safety_preservation'].append(item.safety_preservation)
        if item.actionability:
            scores[item.model][item.target_language]['actionability'].append(item.actionability)

    # Calculate means
    results = {}
    for model, langs in scores.items():
        results[model] = {}
        for lang, metrics in langs.items():
            results[model][lang] = {}
            for metric, values in metrics.items():
                if values:
                    results[model][lang][metric] = {
                        'mean': statistics.mean(values),
                        'n': len(values)
                    }

    return results


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_all_evaluation_materials(translations_file: str = "all_results.json"):
    """
    Generate all human evaluation materials.
    """
    # Generate evaluation items
    items = generate_evaluation_items(translations_file)

    if not items:
        logger.error("No translation results found")
        return

    # Get unique languages
    languages = list(set(item.target_language for item in items))

    print("\n" + "="*60)
    print("GENERATING HUMAN EVALUATION MATERIALS")
    print("="*60)

    # Generate HTML packets for each language
    print("\nGenerating HTML evaluation packets...")
    for lang in languages:
        lang_name = LANGUAGES.get(lang, {}).get('name', lang)
        filepath = generate_reviewer_packet_html(items, lang, reviewer_name=f"{lang_name} Reviewer")
        if filepath:
            print(f"  ✓ {lang_name}: {filepath}")

    # Generate Excel sheets
    print("\nGenerating Excel evaluation sheets...")

    # One sheet per language
    for lang in languages:
        lang_name = LANGUAGES.get(lang, {}).get('name', lang)
        filepath = generate_excel_evaluation_sheet(items, lang)
        if filepath:
            print(f"  ✓ {lang_name}: {filepath}")

    # Combined sheet
    filepath = generate_excel_evaluation_sheet(items, language=None)
    if filepath:
        print(f"  ✓ All languages: {filepath}")

    # Save items for later use
    items_file = HUMAN_EVAL_DIR / "evaluation_items.json"
    with open(items_file, 'w', encoding='utf-8') as f:
        json.dump([item.to_dict() for item in items], f, indent=2)
    print(f"\n✓ Saved evaluation items to: {items_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total evaluation items: {len(items)}")
    print(f"Languages: {len(languages)}")
    for lang in languages:
        lang_name = LANGUAGES.get(lang, {}).get('name', lang)
        count = len([i for i in items if i.target_language == lang])
        print(f"  - {lang_name}: {count} items")

    print("\n" + "="*60)
    print("NEXT STEPS FOR BILINGUAL REVIEWERS")
    print("="*60)
    print("""
1. Send the HTML packets OR Excel sheets to bilingual reviewers
2. Each reviewer evaluates translations in their language
3. Reviewers return completed Excel sheets
4. Run: python human_evaluation.py --aggregate [file1.xlsx] [file2.xlsx] ...
    """)

    return items


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        # Generate evaluation materials
        translations_file = sys.argv[2] if len(sys.argv) > 2 else "all_results.json"
        generate_all_evaluation_materials(translations_file)

    elif len(sys.argv) > 1 and sys.argv[1] == "--aggregate":
        # Aggregate completed evaluations
        if len(sys.argv) < 3:
            print("Usage: python human_evaluation.py --aggregate [file1.xlsx] [file2.xlsx] ...")
            sys.exit(1)

        all_items = []
        for filepath in sys.argv[2:]:
            items = load_completed_evaluations(filepath)
            all_items.extend(items)

        scores = calculate_human_scores(all_items)

        print("\n" + "="*60)
        print("HUMAN EVALUATION SCORES")
        print("="*60)

        for model, langs in scores.items():
            print(f"\n{model}:")
            for lang, metrics in langs.items():
                lang_name = LANGUAGES.get(lang, {}).get('name', lang)
                print(f"  {lang_name}:")
                for metric, data in metrics.items():
                    print(f"    {metric}: {data['mean']:.2f} (n={data['n']})")

    else:
        print("""
Human Evaluation Tools
======================

Usage:
  python human_evaluation.py --generate [translations_file]
      Generate evaluation materials (HTML packets, Excel sheets)

  python human_evaluation.py --aggregate [file1.xlsx] [file2.xlsx] ...
      Aggregate completed evaluations from reviewers

The --generate command creates:
  - HTML evaluation packets for each language (send to reviewers)
  - Excel sheets for data collection
  - JSON file with all evaluation items

After reviewers complete their evaluations:
  1. Collect the completed Excel files
  2. Run --aggregate to combine scores
  3. Merge with automated metrics for final report
        """)
