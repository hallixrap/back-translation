"""
Back Translation Project - NLP Metrics Evaluation
Stanford Clinical Translation Evaluation Framework

This module calculates all automated NLP metrics for translation quality assessment.

Metrics implemented:
- BLEU (Bilingual Evaluation Understudy)
- ChrF (Character n-gram F-score)
- BERTScore (BERT-based semantic similarity)
- COMET (Crosslingual Optimized Metric for Evaluation)
- LaBSE (Language-agnostic BERT Sentence Embedding)
- Readability metrics
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
import traceback

import numpy as np

from config import (
    METRIC_THRESHOLDS, METRICS_DIR, TRANSLATIONS_DIR,
    SUITABILITY_THRESHOLDS, logger
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MetricScores:
    """Stores all metric scores for a single translation."""
    doc_id: str
    model: str
    target_language: str

    # Core metrics (back-translation comparison)
    bleu: Optional[float] = None
    chrf: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None

    # Semantic similarity metrics
    labse_similarity: Optional[float] = None

    # COMET (if available)
    comet_score: Optional[float] = None

    # Readability (English back-translation)
    flesch_reading_ease: Optional[float] = None
    flesch_kincaid_grade: Optional[float] = None

    # Composite scores
    composite_score: Optional[float] = None
    suitability_rating: Optional[str] = None  # "suitable", "caution", "not_recommended"

    # Error tracking
    errors: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


# =============================================================================
# METRIC CALCULATORS - LAZY LOADING
# =============================================================================

# These will be initialized on first use to avoid slow imports
_bleu_scorer = None
_chrf_scorer = None
_bert_scorer = None
_comet_model = None
_labse_model = None


def get_bleu_scorer():
    """Lazy load BLEU scorer."""
    global _bleu_scorer
    if _bleu_scorer is None:
        from sacrebleu.metrics import BLEU
        _bleu_scorer = BLEU(effective_order=True)
        logger.info("BLEU scorer initialized")
    return _bleu_scorer


def get_chrf_scorer():
    """Lazy load ChrF scorer."""
    global _chrf_scorer
    if _chrf_scorer is None:
        from sacrebleu.metrics import CHRF
        _chrf_scorer = CHRF()
        logger.info("ChrF scorer initialized")
    return _chrf_scorer


def get_bert_scorer():
    """Lazy load BERTScore."""
    global _bert_scorer
    if _bert_scorer is None:
        # BERTScore is imported on demand
        import bert_score
        _bert_scorer = bert_score
        logger.info("BERTScore initialized")
    return _bert_scorer


def get_labse_model():
    """Lazy load LaBSE model for multilingual similarity."""
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        _labse_model = SentenceTransformer('sentence-transformers/LaBSE')
        logger.info("LaBSE model initialized")
    return _labse_model


def get_comet_model():
    """Lazy load COMET model."""
    global _comet_model
    if _comet_model is None:
        try:
            from comet import download_model, load_from_checkpoint
            # Download the default model if not present
            model_path = download_model("Unbabel/wmt22-comet-da")
            _comet_model = load_from_checkpoint(model_path)
            logger.info("COMET model initialized")
        except Exception as e:
            logger.warning(f"COMET model not available: {e}")
            _comet_model = "unavailable"
    return _comet_model


# =============================================================================
# INDIVIDUAL METRIC CALCULATIONS
# =============================================================================

def calculate_bleu(hypothesis: str, reference: str) -> float:
    """
    Calculate BLEU score between hypothesis and reference.

    Args:
        hypothesis: The text to evaluate (back-translation)
        reference: The reference text (original English)

    Returns:
        BLEU score (0-100)
    """
    scorer = get_bleu_scorer()
    # sacrebleu expects list of hypotheses and list of list of references
    score = scorer.corpus_score([hypothesis], [[reference]])
    return score.score


def calculate_chrf(hypothesis: str, reference: str) -> float:
    """
    Calculate ChrF (Character n-gram F-score).

    More robust than BLEU for morphologically rich languages.

    Returns:
        ChrF score (0-100)
    """
    scorer = get_chrf_scorer()
    score = scorer.corpus_score([hypothesis], [[reference]])
    return score.score


def calculate_bertscore(hypothesis: str, reference: str, lang: str = "en") -> dict:
    """
    Calculate BERTScore for semantic similarity.

    Returns:
        Dictionary with precision, recall, and F1 scores (0-1)
    """
    bert_score = get_bert_scorer()
    P, R, F1 = bert_score.score(
        [hypothesis],
        [reference],
        lang=lang,
        verbose=False
    )
    return {
        "precision": P.item(),
        "recall": R.item(),
        "f1": F1.item()
    }


def calculate_labse_similarity(text1: str, text2: str) -> float:
    """
    Calculate LaBSE cosine similarity between two texts.

    LaBSE works across languages, so can compare original English
    directly with target language translation.

    Returns:
        Cosine similarity score (0-1)
    """
    model = get_labse_model()

    # Encode both texts
    embeddings = model.encode([text1, text2])

    # Calculate cosine similarity
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )

    return float(similarity)


def calculate_comet(source: str, translation: str, reference: str = None) -> Optional[float]:
    """
    Calculate COMET score.

    COMET is a neural metric trained on human judgments.

    Args:
        source: Source text (original English)
        translation: Translation to evaluate
        reference: Reference translation (optional for QE models)

    Returns:
        COMET score (typically 0-1, higher is better)
    """
    model = get_comet_model()

    if model == "unavailable":
        return None

    try:
        data = [{"src": source, "mt": translation, "ref": reference or source}]
        # Use CPU accelerator to avoid multiprocessing issues
        output = model.predict(data, batch_size=1, accelerator='cpu', progress_bar=False)
        return output.scores[0]
    except Exception as e:
        logger.warning(f"COMET calculation failed: {e}")
        return None


def calculate_readability(text: str) -> dict:
    """
    Calculate readability metrics for English text.

    Returns:
        Dictionary with Flesch Reading Ease and Flesch-Kincaid Grade Level
    """
    try:
        import textstat
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text)
        }
    except ImportError:
        logger.warning("textstat not installed. Run: pip install textstat")
        return {"flesch_reading_ease": None, "flesch_kincaid_grade": None}
    except Exception as e:
        logger.warning(f"Readability calculation failed: {e}")
        return {"flesch_reading_ease": None, "flesch_kincaid_grade": None}


# =============================================================================
# COMPREHENSIVE METRIC EVALUATION
# =============================================================================

def evaluate_translation(
    doc_id: str,
    model: str,
    target_language: str,
    original_english: str,
    translated_text: str,
    back_translated_text: str,
    include_comet: bool = True,
    include_labse_crosslingual: bool = True
) -> MetricScores:
    """
    Calculate all metrics for a single translation.

    Args:
        doc_id: Document identifier
        model: Model name
        target_language: Target language key
        original_english: Original English source text
        translated_text: Translation in target language
        back_translated_text: Back-translation to English

    Returns:
        MetricScores object with all calculated metrics
    """
    scores = MetricScores(
        doc_id=doc_id,
        model=model,
        target_language=target_language
    )

    logger.info(f"Evaluating: {doc_id} | {model} | {target_language}")

    # --- Back-Translation Metrics (compare back-translation to original) ---

    # BLEU
    try:
        scores.bleu = calculate_bleu(back_translated_text, original_english)
        logger.debug(f"  BLEU: {scores.bleu:.2f}")
    except Exception as e:
        scores.errors.append(f"BLEU: {str(e)}")
        logger.warning(f"  BLEU failed: {e}")

    # ChrF
    try:
        scores.chrf = calculate_chrf(back_translated_text, original_english)
        logger.debug(f"  ChrF: {scores.chrf:.2f}")
    except Exception as e:
        scores.errors.append(f"ChrF: {str(e)}")
        logger.warning(f"  ChrF failed: {e}")

    # BERTScore
    try:
        bert_scores = calculate_bertscore(back_translated_text, original_english)
        scores.bertscore_precision = bert_scores["precision"]
        scores.bertscore_recall = bert_scores["recall"]
        scores.bertscore_f1 = bert_scores["f1"]
        logger.debug(f"  BERTScore F1: {scores.bertscore_f1:.4f}")
    except Exception as e:
        scores.errors.append(f"BERTScore: {str(e)}")
        logger.warning(f"  BERTScore failed: {e}")

    # --- Cross-lingual Metrics (compare translation directly to original) ---

    # LaBSE (cross-lingual)
    if include_labse_crosslingual:
        try:
            scores.labse_similarity = calculate_labse_similarity(
                original_english, translated_text
            )
            logger.debug(f"  LaBSE: {scores.labse_similarity:.4f}")
        except Exception as e:
            scores.errors.append(f"LaBSE: {str(e)}")
            logger.warning(f"  LaBSE failed: {e}")

    # COMET
    if include_comet:
        try:
            scores.comet_score = calculate_comet(
                source=original_english,
                translation=back_translated_text,
                reference=original_english
            )
            if scores.comet_score:
                logger.debug(f"  COMET: {scores.comet_score:.4f}")
        except Exception as e:
            scores.errors.append(f"COMET: {str(e)}")
            logger.warning(f"  COMET failed: {e}")

    # --- Readability Metrics (on back-translation) ---
    try:
        readability = calculate_readability(back_translated_text)
        scores.flesch_reading_ease = readability["flesch_reading_ease"]
        scores.flesch_kincaid_grade = readability["flesch_kincaid_grade"]
    except Exception as e:
        scores.errors.append(f"Readability: {str(e)}")
        logger.warning(f"  Readability failed: {e}")

    # --- Calculate Composite Score and Suitability ---
    scores.composite_score = calculate_composite_score(scores)
    scores.suitability_rating = determine_suitability(scores.composite_score)

    logger.info(f"  Composite: {scores.composite_score:.3f} ({scores.suitability_rating})")

    return scores


def calculate_composite_score(scores: MetricScores) -> float:
    """
    Calculate a composite score from all available metrics.

    Weights are based on metric reliability for clinical translation:
    - BERTScore F1: 30% (best semantic similarity measure)
    - COMET: 25% (trained on human judgments)
    - LaBSE: 20% (cross-lingual capability)
    - ChrF: 15% (character-level robustness)
    - BLEU: 10% (traditional baseline)

    Returns:
        Composite score (0-1)
    """
    weights = {
        "bertscore_f1": 0.30,
        "comet": 0.25,
        "labse": 0.20,
        "chrf": 0.15,
        "bleu": 0.10
    }

    weighted_sum = 0.0
    total_weight = 0.0

    # BERTScore F1 (already 0-1)
    if scores.bertscore_f1 is not None:
        weighted_sum += scores.bertscore_f1 * weights["bertscore_f1"]
        total_weight += weights["bertscore_f1"]

    # COMET (typically 0-1)
    if scores.comet_score is not None:
        weighted_sum += scores.comet_score * weights["comet"]
        total_weight += weights["comet"]

    # LaBSE (0-1)
    if scores.labse_similarity is not None:
        weighted_sum += scores.labse_similarity * weights["labse"]
        total_weight += weights["labse"]

    # ChrF (0-100, normalize to 0-1)
    if scores.chrf is not None:
        weighted_sum += (scores.chrf / 100) * weights["chrf"]
        total_weight += weights["chrf"]

    # BLEU (0-100, normalize to 0-1)
    if scores.bleu is not None:
        weighted_sum += (scores.bleu / 100) * weights["bleu"]
        total_weight += weights["bleu"]

    # Return weighted average, or 0 if no metrics available
    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.0


def determine_suitability(composite_score: float) -> str:
    """
    Determine clinical suitability rating based on composite score.

    Returns:
        "suitable", "caution", or "not_recommended"
    """
    if composite_score >= SUITABILITY_THRESHOLDS["suitable"]:
        return "suitable"
    elif composite_score >= SUITABILITY_THRESHOLDS["caution"]:
        return "caution"
    else:
        return "not_recommended"


# =============================================================================
# BATCH EVALUATION
# =============================================================================

def evaluate_all_translations(
    translations_file: str = "all_results.json",
    output_file: str = "all_metrics.json",
    include_comet: bool = True
) -> list[MetricScores]:
    """
    Evaluate all translations from a results file.

    Args:
        translations_file: Input file with translation results
        output_file: Output file for metrics
        include_comet: Whether to include COMET (slower)

    Returns:
        List of MetricScores objects
    """
    from translation_pipeline import TranslationResult

    # Load translations
    filepath = TRANSLATIONS_DIR / translations_file
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    translations = [TranslationResult.from_dict(d) for d in data]

    logger.info(f"Evaluating {len(translations)} translations")

    # Filter to successful translations only
    translations = [t for t in translations if t.success and t.back_translated_text]
    logger.info(f"  {len(translations)} successful translations to evaluate")

    all_scores = []

    for i, trans in enumerate(translations):
        try:
            scores = evaluate_translation(
                doc_id=trans.doc_id,
                model=trans.model,
                target_language=trans.target_language,
                original_english=trans.original_text,
                translated_text=trans.translated_text,
                back_translated_text=trans.back_translated_text,
                include_comet=include_comet
            )
            all_scores.append(scores)

        except Exception as e:
            logger.error(f"Evaluation failed for {trans.doc_id}/{trans.model}/{trans.target_language}: {e}")
            logger.error(traceback.format_exc())

        # Progress logging
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(translations)}")

    # Save results
    save_metrics(all_scores, output_file)

    return all_scores


def save_metrics(scores: list[MetricScores], filename: str):
    """Save metrics to JSON file."""
    filepath = METRICS_DIR / filename
    data = [s.to_dict() for s in scores]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(scores)} metric scores to {filepath}")
    return filepath


def load_metrics(filename: str) -> list[MetricScores]:
    """Load metrics from JSON file."""
    filepath = METRICS_DIR / filename

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scores = [MetricScores.from_dict(d) for d in data]
    logger.info(f"Loaded {len(scores)} metric scores from {filepath}")
    return scores


# =============================================================================
# QUICK METRIC FUNCTIONS (for testing)
# =============================================================================

def quick_evaluate(original: str, back_translation: str) -> dict:
    """
    Quick evaluation of a single back-translation pair.
    Useful for testing and debugging.

    Returns:
        Dictionary with core metrics
    """
    results = {}

    # BLEU
    try:
        results["bleu"] = calculate_bleu(back_translation, original)
    except:
        results["bleu"] = None

    # ChrF
    try:
        results["chrf"] = calculate_chrf(back_translation, original)
    except:
        results["chrf"] = None

    # BERTScore
    try:
        bert = calculate_bertscore(back_translation, original)
        results["bertscore_f1"] = bert["f1"]
    except:
        results["bertscore_f1"] = None

    return results


# =============================================================================
# METRIC INTERPRETATION HELPERS
# =============================================================================

def interpret_bleu(score: float) -> str:
    """Interpret BLEU score for clinical context."""
    if score >= 40:
        return "Excellent - High lexical overlap with original"
    elif score >= 30:
        return "Good - Substantial lexical similarity"
    elif score >= 20:
        return "Acceptable - Moderate similarity, review recommended"
    else:
        return "Poor - Low similarity, significant content may be lost"


def interpret_bertscore(score: float) -> str:
    """Interpret BERTScore for clinical context."""
    if score >= 0.90:
        return "Excellent - Strong semantic preservation"
    elif score >= 0.85:
        return "Good - Core meaning well preserved"
    elif score >= 0.80:
        return "Acceptable - Most meaning preserved, some nuance may be lost"
    else:
        return "Concerning - Potential meaning loss, human review essential"


def interpret_composite(score: float) -> str:
    """Interpret composite score for clinical recommendations."""
    if score >= 0.80:
        return "SUITABLE for clinical use with standard review"
    elif score >= 0.60:
        return "USE WITH CAUTION - Enhanced human review recommended"
    else:
        return "NOT RECOMMENDED for clinical use without full professional review"


def print_metric_summary(scores: MetricScores):
    """Print a formatted summary of metric scores."""
    print(f"\n{'='*60}")
    print(f"METRICS: {scores.doc_id} | {scores.model} | {scores.target_language}")
    print(f"{'='*60}")

    print(f"\nBack-Translation Metrics:")
    print(f"  BLEU:        {scores.bleu:.2f if scores.bleu else 'N/A'}")
    print(f"  ChrF:        {scores.chrf:.2f if scores.chrf else 'N/A'}")
    print(f"  BERTScore:   {scores.bertscore_f1:.4f if scores.bertscore_f1 else 'N/A'}")

    print(f"\nSemantic Metrics:")
    print(f"  LaBSE:       {scores.labse_similarity:.4f if scores.labse_similarity else 'N/A'}")
    print(f"  COMET:       {scores.comet_score:.4f if scores.comet_score else 'N/A'}")

    print(f"\nReadability (Back-translation):")
    print(f"  Flesch Ease: {scores.flesch_reading_ease:.1f if scores.flesch_reading_ease else 'N/A'}")
    print(f"  Grade Level: {scores.flesch_kincaid_grade:.1f if scores.flesch_kincaid_grade else 'N/A'}")

    print(f"\n{'='*60}")
    print(f"COMPOSITE SCORE: {scores.composite_score:.3f}")
    print(f"SUITABILITY:     {scores.suitability_rating.upper()}")
    print(f"{'='*60}")
    print(f"\nInterpretation: {interpret_composite(scores.composite_score)}")

    if scores.errors:
        print(f"\nWarnings: {', '.join(scores.errors)}")


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test with sample texts
        original = """Heart Failure: What You Need to Know

What is Heart Failure?
Heart failure means your heart is not pumping blood as well as it should.

Warning Signs - Call Your Doctor If You Have:
• Sudden weight gain
• Increased swelling in your legs
• Shortness of breath"""

        back_translation = """Heart Failure: What You Should Know

What is Heart Failure?
Heart failure means that your heart is not pumping blood as efficiently as it should.

Warning Signs - Contact Your Doctor If You Experience:
• Sudden weight gain
• Increased swelling in your legs
• Difficulty breathing"""

        print("Testing metrics with sample text...\n")
        results = quick_evaluate(original, back_translation)

        print(f"BLEU: {results['bleu']:.2f}")
        print(f"ChrF: {results['chrf']:.2f}")
        print(f"BERTScore F1: {results['bertscore_f1']:.4f}")

    elif len(sys.argv) > 1 and sys.argv[1] == "--evaluate":
        # Run full evaluation on existing translations
        input_file = sys.argv[2] if len(sys.argv) > 2 else "all_results.json"
        evaluate_all_translations(input_file)

    else:
        print("""
Usage:
  python metrics_evaluation.py --test           # Test metrics with sample text
  python metrics_evaluation.py --evaluate [file]  # Evaluate translations file

The evaluation will calculate:
  - BLEU (lexical overlap)
  - ChrF (character n-gram)
  - BERTScore (semantic similarity)
  - LaBSE (cross-lingual similarity)
  - COMET (neural metric)
  - Readability scores
  - Composite suitability score
        """)
