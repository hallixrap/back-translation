#!/usr/bin/env python3
"""
Back Translation Project - Main Orchestration Script
Stanford Clinical Translation Evaluation Framework

This is the main entry point for running the complete translation evaluation pipeline.

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --setup            # Verify setup and dependencies
    python run_pipeline.py --test             # Run a quick test with one document
    python run_pipeline.py --translate        # Run translations only
    python run_pipeline.py --evaluate         # Run metrics evaluation only
    python run_pipeline.py --report           # Generate reports only
    python run_pipeline.py --human-eval       # Generate human evaluation materials
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from config import (
    API_KEYS, ACTIVE_MODELS, ACTIVE_LANGUAGES, LANGUAGES,
    OUTPUT_DIR, TRANSLATIONS_DIR, METRICS_DIR, REPORTS_DIR,
    logger
)


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║      🔬 STANFORD BACK TRANSLATION EVALUATION FRAMEWORK                       ║
║                                                                              ║
║      Clinical Translation Quality Assessment for Patient Education           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n📦 Checking dependencies...")

    required = {
        'openai': 'OpenAI API client',
        'anthropic': 'Anthropic API client',
        'google.generativeai': 'Google Generative AI',
        'sacrebleu': 'BLEU/ChrF metrics',
        'bert_score': 'BERTScore metric',
        'sentence_transformers': 'LaBSE embeddings',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical operations',
    }

    optional = {
        'comet': 'COMET metric (optional)',
        'matplotlib': 'Plotting (optional)',
        'seaborn': 'Enhanced plotting (optional)',
        'textstat': 'Readability metrics (optional)',
        'openpyxl': 'Excel export (optional)',
    }

    missing_required = []
    missing_optional = []

    for module, description in required.items():
        try:
            __import__(module.split('.')[0])
            print(f"  ✓ {description}")
        except ImportError:
            print(f"  ✗ {description} - MISSING")
            missing_required.append(module)

    print("\n  Optional packages:")
    for module, description in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {description}")
        except ImportError:
            print(f"  ○ {description} - not installed")
            missing_optional.append(module)

    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False

    print("\n✅ All required dependencies installed!")
    return True


def check_api_keys():
    """Check if API keys are configured."""
    print("\n🔑 Checking API keys...")

    all_configured = True

    for provider, key in API_KEYS.items():
        if key and not key.startswith("YOUR_"):
            # Check if key looks valid (basic check)
            if len(key) > 10:
                print(f"  ✓ {provider.capitalize()}: Configured")
            else:
                print(f"  ⚠ {provider.capitalize()}: Key looks too short")
                all_configured = False
        else:
            print(f"  ✗ {provider.capitalize()}: Not configured")
            all_configured = False

    if not all_configured:
        print("\n⚠️  Some API keys are not configured.")
        print("   Edit scripts/config.py or set environment variables:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY")

    return all_configured


def check_directories():
    """Ensure all directories exist."""
    print("\n📁 Checking directories...")

    for dir_path in [OUTPUT_DIR, TRANSLATIONS_DIR, METRICS_DIR, REPORTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")

    return True


def run_setup():
    """Run setup verification."""
    print_banner()
    print("=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)

    deps_ok = check_dependencies()
    keys_ok = check_api_keys()
    dirs_ok = check_directories()

    print("\n" + "=" * 60)
    if deps_ok and dirs_ok:
        if keys_ok:
            print("✅ Setup complete! Ready to run pipeline.")
        else:
            print("⚠️  Setup mostly complete. Configure API keys to enable all models.")
    else:
        print("❌ Setup incomplete. Please install missing dependencies.")
    print("=" * 60)

    return deps_ok and dirs_ok


def run_quick_test():
    """Run a quick test with one document, one language, one model."""
    print_banner()
    print("=" * 60)
    print("QUICK TEST MODE")
    print("=" * 60)

    from document_sources import SAMPLE_DOCUMENTS
    from translation_pipeline import run_translation_pipeline, verify_api_connections
    from metrics_evaluation import evaluate_translation, print_metric_summary

    # First verify API connections
    print("\n🔌 Verifying API connections...")
    status = verify_api_connections()

    # Find a working model
    working_model = None
    for model in ACTIVE_MODELS:
        if status.get(model, "").startswith("✓"):
            working_model = model
            break

    if not working_model:
        print("❌ No working models found. Check your API keys.")
        return False

    print(f"\n✅ Using model: {working_model}")

    # Use first document and Spanish
    test_doc = SAMPLE_DOCUMENTS[0]
    test_lang = "spanish"

    print(f"\n📄 Test document: {test_doc.title}")
    print(f"🌐 Target language: {LANGUAGES[test_lang]['name']}")
    print(f"🤖 Model: {working_model}")

    # Run translation
    print("\n" + "-" * 40)
    print("Running translation pipeline...")
    print("-" * 40)

    start_time = time.time()
    result = run_translation_pipeline(
        doc_id=test_doc.doc_id,
        english_text=test_doc.english_text[:1000],  # Truncate for speed
        target_language_key=test_lang,
        model_name=working_model
    )
    translation_time = time.time() - start_time

    if not result.success:
        print(f"❌ Translation failed: {result.error_message}")
        return False

    print(f"✅ Translation completed in {translation_time:.2f}s")

    # Preview translations
    print("\n" + "-" * 40)
    print("Translation Preview")
    print("-" * 40)
    print(f"\n📝 Original (first 300 chars):\n{result.original_text[:300]}...")
    print(f"\n🌐 Translated (first 300 chars):\n{result.translated_text[:300]}...")
    print(f"\n🔄 Back-translated (first 300 chars):\n{result.back_translated_text[:300]}...")

    # Run metrics
    print("\n" + "-" * 40)
    print("Calculating metrics...")
    print("-" * 40)

    scores = evaluate_translation(
        doc_id=result.doc_id,
        model=result.model,
        target_language=result.target_language,
        original_english=result.original_text,
        translated_text=result.translated_text,
        back_translated_text=result.back_translated_text,
        include_comet=False  # Skip COMET for speed
    )

    print_metric_summary(scores)

    print("\n" + "=" * 60)
    print("✅ QUICK TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nThe pipeline is working. Run 'python run_pipeline.py' for full execution.")

    return True


def run_translations():
    """Run the translation pipeline on all documents."""
    print("\n" + "=" * 60)
    print("RUNNING TRANSLATIONS")
    print("=" * 60)

    from document_sources import initialize_documents
    from translation_pipeline import process_batch_with_resume, save_results

    # Load documents
    documents = initialize_documents()
    print(f"\n📄 Loaded {len(documents)} documents")
    print(f"🌐 Languages: {[LANGUAGES[l]['name'] for l in ACTIVE_LANGUAGES]}")
    print(f"🤖 Models: {ACTIVE_MODELS}")

    total_combos = len(documents) * len(ACTIVE_LANGUAGES) * len(ACTIVE_MODELS)
    print(f"\n📊 Total combinations: {total_combos}")

    # Estimate time
    est_seconds = total_combos * 5  # ~5 seconds per combination
    est_minutes = est_seconds / 60
    print(f"⏱️  Estimated time: {est_minutes:.0f} minutes")

    confirm = input("\nProceed with translations? [y/N]: ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        return None

    # Run translations with resume capability
    print("\n" + "-" * 40)
    start_time = time.time()

    results = process_batch_with_resume(
        documents=documents,
        languages=ACTIVE_LANGUAGES,
        models=ACTIVE_MODELS
    )

    elapsed = time.time() - start_time
    print(f"\n✅ Translations completed in {elapsed/60:.1f} minutes")
    print(f"📁 Results saved to: {TRANSLATIONS_DIR / 'all_results.json'}")

    # Summary
    successful = len([r for r in results if r.success])
    failed = len([r for r in results if not r.success])
    print(f"\n📊 Summary: {successful} successful, {failed} failed")

    return results


def run_evaluation():
    """Run metrics evaluation on translations."""
    print("\n" + "=" * 60)
    print("RUNNING METRICS EVALUATION")
    print("=" * 60)

    from metrics_evaluation import evaluate_all_translations

    # Check if translations exist
    translations_file = TRANSLATIONS_DIR / "all_results.json"
    if not translations_file.exists():
        print(f"❌ No translations found at {translations_file}")
        print("Run translations first with: python run_pipeline.py --translate")
        return None

    print(f"\n📄 Loading translations from: {translations_file}")

    # Run evaluation
    scores = evaluate_all_translations(
        translations_file="all_results.json",
        output_file="all_metrics.json",
        include_comet=True
    )

    print(f"\n✅ Evaluation completed")
    print(f"📁 Metrics saved to: {METRICS_DIR / 'all_metrics.json'}")

    return scores


def run_reports():
    """Generate all reports."""
    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60)

    from results_aggregation import generate_all_reports

    # Check if metrics exist
    metrics_file = METRICS_DIR / "all_metrics.json"
    if not metrics_file.exists():
        print(f"❌ No metrics found at {metrics_file}")
        print("Run evaluation first with: python run_pipeline.py --evaluate")
        return None

    # Generate reports
    report = generate_all_reports("all_metrics.json")

    print(f"\n✅ Reports generated")
    print(f"📁 Reports saved to: {REPORTS_DIR}")

    return report


def run_human_eval_generation():
    """Generate human evaluation materials."""
    print("\n" + "=" * 60)
    print("GENERATING HUMAN EVALUATION MATERIALS")
    print("=" * 60)

    from human_evaluation import generate_all_evaluation_materials

    # Check if translations exist
    translations_file = TRANSLATIONS_DIR / "all_results.json"
    if not translations_file.exists():
        print(f"❌ No translations found at {translations_file}")
        print("Run translations first with: python run_pipeline.py --translate")
        return None

    # Generate materials
    items = generate_all_evaluation_materials("all_results.json")

    return items


def run_full_pipeline():
    """Run the complete pipeline."""
    print_banner()

    # Setup check
    if not run_setup():
        return

    print("\n" + "=" * 60)
    print("FULL PIPELINE EXECUTION")
    print("=" * 60)

    pipeline_start = time.time()

    # Step 1: Translations
    print("\n📌 STEP 1/4: Translations")
    results = run_translations()
    if results is None:
        return

    # Step 2: Metrics Evaluation
    print("\n📌 STEP 2/4: Metrics Evaluation")
    scores = run_evaluation()

    # Step 3: Report Generation
    print("\n📌 STEP 3/4: Report Generation")
    report = run_reports()

    # Step 4: Human Evaluation Materials
    print("\n📌 STEP 4/4: Human Evaluation Materials")
    eval_items = run_human_eval_generation()

    # Final summary
    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED")
    print("=" * 60)
    print(f"\n⏱️  Total time: {pipeline_elapsed/60:.1f} minutes")
    print(f"\n📁 Output files:")
    print(f"   - Translations: {TRANSLATIONS_DIR / 'all_results.json'}")
    print(f"   - Metrics: {METRICS_DIR / 'all_metrics.json'}")
    print(f"   - Reports: {REPORTS_DIR}")
    print(f"   - Kevin's Scorecard: {REPORTS_DIR / 'kevin_scorecard.txt'}")
    print(f"   - Human Eval Packets: {REPORTS_DIR / 'eval_packet_*.html'}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Review kevin_scorecard.txt for the main deliverable
2. Send HTML evaluation packets to bilingual reviewers
3. Collect completed evaluation Excel sheets
4. Run: python human_evaluation.py --aggregate [files...]
5. Merge human scores with automated metrics for final report
    """)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # Default: run full pipeline
        run_full_pipeline()

    elif sys.argv[1] == "--setup":
        run_setup()

    elif sys.argv[1] == "--test":
        if not run_setup():
            return
        run_quick_test()

    elif sys.argv[1] == "--translate":
        print_banner()
        run_translations()

    elif sys.argv[1] == "--evaluate":
        print_banner()
        run_evaluation()

    elif sys.argv[1] == "--report":
        print_banner()
        run_reports()

    elif sys.argv[1] == "--human-eval":
        print_banner()
        run_human_eval_generation()

    elif sys.argv[1] in ["--help", "-h"]:
        print(__doc__)

    else:
        print(f"Unknown option: {sys.argv[1]}")
        print("Use --help for usage information.")


if __name__ == "__main__":
    main()
