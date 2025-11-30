"""
Back Translation Project - Configuration
Stanford Clinical Translation Evaluation Framework

This file contains all configuration settings, API keys, and constants.
IMPORTANT: Replace placeholder API keys with your actual keys before running.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DOCUMENTS_DIR = DATA_DIR / "source_documents"
TRANSLATIONS_DIR = OUTPUT_DIR / "translations"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, DOCUMENTS_DIR, TRANSLATIONS_DIR, METRICS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API KEYS - REPLACE WITH YOUR ACTUAL KEYS
# =============================================================================

# Option 1: Set as environment variables (recommended for security)
# export OPENAI_API_KEY="sk-..."
# export ANTHROPIC_API_KEY="sk-ant-..."
# export GOOGLE_API_KEY="..."

# Option 2: Direct assignment (less secure, but convenient for testing)
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY"),
    "moonshot": os.getenv("MOONSHOT_API_KEY", "YOUR_MOONSHOT_API_KEY"),  # Kimi K2 direct from platform.moonshot.ai
}

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODELS = {
    # ==========================================================================
    # FRONTIER MODELS - November 2025
    # ==========================================================================
    "gpt-5.1": {
        "provider": "openai",
        "model_id": "gpt-5.1",  # OpenAI flagship - released Nov 2025
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    "claude-opus-4.5": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5-20251101",  # Anthropic flagship - released Nov 24, 2025
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    "claude-sonnet-4.5": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5-20250929",
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    "claude-sonnet-4": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    "gemini-3-pro": {
        "provider": "google",
        "model_id": "gemini-3-pro-preview",  # Google flagship - released Nov 2025
        "max_tokens": 16384,  # Increased for longer medical documents
        "temperature": 0.3,
    },
    "kimi-k2": {
        "provider": "moonshot",
        "model_id": "kimi-k2-thinking",  # Kimi K2 Thinking via Moonshot direct API
        "max_tokens": 32768,  # Thinking models need higher limit (reasoning + content share this budget)
        "temperature": 0.3,
    },
}

# =============================================================================
# ACTIVE MODELS - November 2025 Frontier Models
# =============================================================================
# These are the models that will be evaluated in the study
ACTIVE_MODELS = [
    # Frontier models (primary evaluation)
    "gpt-5.1",            # OpenAI flagship (Nov 2025)
    "claude-opus-4.5",    # Anthropic flagship (Nov 24, 2025)
    "gemini-3-pro",       # Google flagship (Nov 25, 2025)
    "kimi-k2",            # Moonshot direct API (platform.moonshot.ai)
]

# =============================================================================
# LANGUAGE CONFIGURATIONS
# =============================================================================

LANGUAGES = {
    "spanish": {
        "code": "es",
        "name": "Spanish",
        "script": "Latin",
        "resource_level": "high",
        "stanford_priority": "very_high",
    },
    "chinese_simplified": {
        "code": "zh-CN",
        "name": "Chinese (Simplified)",
        "script": "Non-Latin",
        "resource_level": "high",
        "stanford_priority": "high",
    },
    "chinese_traditional": {
        "code": "zh-TW",
        "name": "Chinese (Traditional)",
        "script": "Non-Latin",
        "resource_level": "high",
        "stanford_priority": "medium",
    },
    "vietnamese": {
        "code": "vi",
        "name": "Vietnamese",
        "script": "Latin (diacritics)",
        "resource_level": "medium",
        "stanford_priority": "high",
    },
    "tagalog": {
        "code": "tl",
        "name": "Tagalog/Filipino",
        "script": "Latin",
        "resource_level": "medium",
        "stanford_priority": "medium",
    },
    "russian": {
        "code": "ru",
        "name": "Russian",
        "script": "Cyrillic",
        "resource_level": "high",
        "stanford_priority": "medium",
    },
    "arabic": {
        "code": "ar",
        "name": "Arabic",
        "script": "Non-Latin (RTL)",
        "resource_level": "medium",
        "stanford_priority": "medium",
    },
    "korean": {
        "code": "ko",
        "name": "Korean",
        "script": "Non-Latin",
        "resource_level": "high",
        "stanford_priority": "medium",
    },
    "haitian_creole": {
        "code": "ht",
        "name": "Haitian Creole",
        "script": "Latin",
        "resource_level": "very_low",  # Key test case for underrepresented languages
        "stanford_priority": "low",
    },
    "portuguese": {
        "code": "pt",
        "name": "Portuguese",
        "script": "Latin",
        "resource_level": "high",
        "stanford_priority": "medium",
    },
    "hindi": {
        "code": "hi",
        "name": "Hindi",
        "script": "Devanagari",
        "resource_level": "medium",
        "stanford_priority": "low",
    },
    "farsi": {
        "code": "fa",
        "name": "Farsi/Persian",
        "script": "Non-Latin (RTL)",
        "resource_level": "medium",
        "stanford_priority": "low",
    },
}

# Languages to actually evaluate (adjust based on bilingual reviewer availability)
ACTIVE_LANGUAGES = [
    "spanish",           # Baseline - Eduardo's existing work
    "chinese_simplified",
    "vietnamese",
    "tagalog",
    "russian",
    "arabic",
    "korean",
    "haitian_creole",    # Key test case for underrepresented
]

# =============================================================================
# DOCUMENT CATEGORIES - BROADER THAN CARDIOLOGY
# =============================================================================

DOCUMENT_CATEGORIES = {
    "cardiology": {
        "description": "Heart and cardiovascular conditions",
        "sources": ["NIH NHLBI", "MedlinePlus", "AHA"],
        "topics": [
            "heart_failure",
            "heart_attack_mi",
            "atrial_fibrillation",
            "hypertension",
            "coronary_artery_disease",
            "cardiac_catheterization",
            "pacemaker_care",
            "anticoagulation",
        ],
    },
    "diabetes": {
        "description": "Diabetes and metabolic conditions",
        "sources": ["NIH NIDDK", "MedlinePlus", "ADA"],
        "topics": [
            "type2_diabetes_basics",
            "insulin_management",
            "blood_sugar_monitoring",
            "diabetic_diet",
            "hypoglycemia",
            "diabetic_foot_care",
        ],
    },
    "respiratory": {
        "description": "Lung and breathing conditions",
        "sources": ["NIH NHLBI", "MedlinePlus"],
        "topics": [
            "asthma_management",
            "copd_care",
            "pneumonia_recovery",
            "inhaler_use",
            "oxygen_therapy",
        ],
    },
    "medications": {
        "description": "Common medication instructions",
        "sources": ["MedlinePlus", "FDA"],
        "topics": [
            "blood_thinners",
            "pain_management",
            "antibiotic_use",
            "medication_safety",
        ],
    },
    "preventive_care": {
        "description": "Preventive health and screenings",
        "sources": ["NIH", "CDC", "MedlinePlus"],
        "topics": [
            "cancer_screening",
            "immunizations",
            "health_checkups",
            "lifestyle_modifications",
        ],
    },
    "emergency_care": {
        "description": "Post-emergency and urgent care instructions",
        "sources": ["MedlinePlus", "Hospital templates"],
        "topics": [
            "chest_pain_followup",
            "stroke_warning_signs",
            "head_injury_monitoring",
            "allergic_reaction",
        ],
    },
    "surgical": {
        "description": "Pre and post-surgical care",
        "sources": ["NIH", "Hospital templates"],
        "topics": [
            "general_surgery_prep",
            "post_surgery_care",
            "wound_care",
            "anesthesia_recovery",
        ],
    },
    "mental_health": {
        "description": "Mental health and wellness",
        "sources": ["NIH NIMH", "MedlinePlus"],
        "topics": [
            "depression_basics",
            "anxiety_management",
            "stress_reduction",
            "sleep_hygiene",
        ],
    },
}

# =============================================================================
# METRIC THRESHOLDS
# =============================================================================

METRIC_THRESHOLDS = {
    "bleu": {
        "excellent": 40,
        "good": 30,
        "acceptable": 20,
        "description": "Bilingual Evaluation Understudy (0-100)",
    },
    "chrf": {
        "excellent": 60,
        "good": 50,
        "acceptable": 40,
        "description": "Character n-gram F-score (0-100)",
    },
    "bertscore": {
        "excellent": 0.90,
        "good": 0.85,
        "acceptable": 0.80,
        "description": "BERT-based semantic similarity (0-1)",
    },
    "comet": {
        "excellent": 0.80,
        "good": 0.70,
        "acceptable": 0.60,
        "description": "Crosslingual Optimized Metric for Evaluation (0-1)",
    },
    "labse": {
        "excellent": 0.85,
        "good": 0.75,
        "acceptable": 0.65,
        "description": "Language-agnostic BERT Sentence Embedding similarity (0-1)",
    },
}

# Overall suitability thresholds (composite score)
SUITABILITY_THRESHOLDS = {
    "suitable": 0.80,       # Green - recommended for clinical use
    "caution": 0.60,        # Yellow - requires human review
    "not_recommended": 0.0,  # Red - not suitable for clinical use
}

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

TRANSLATION_SYSTEM_PROMPT = """You are an expert medical translator specializing in patient education materials.
Your translations must:
1. Preserve ALL medical terminology accurately
2. Maintain patient-friendly readability (aim for 6th-8th grade reading level)
3. Be culturally appropriate for the target language speakers
4. Keep ALL safety warnings and critical information intact
5. Preserve document structure (headings, bullet points, numbered lists)

CRITICAL: Never omit, modify, or soften any medical warnings or safety information."""

TRANSLATION_USER_PROMPT = """Translate the following patient education document from English to {target_language}.

Requirements:
- Maintain exact document structure and formatting
- Preserve all medical terms accurately (use commonly understood equivalents when available)
- Keep all numbered lists, bullet points, and section headers
- Do not add explanations or commentary
- Do not remove any content

Document to translate:
---
{document_text}
---

Provide only the {target_language} translation, nothing else."""

BACK_TRANSLATION_SYSTEM_PROMPT = """You are a professional medical translator. Your task is to translate text back to English.

CRITICAL INSTRUCTIONS:
1. Translate EXACTLY what is written - do not correct perceived errors
2. Preserve the meaning and structure as closely as possible
3. If something seems unclear or wrong in the source, translate it literally anyway
4. Do not add any explanations or notes about the translation"""

BACK_TRANSLATION_USER_PROMPT = """Translate the following {source_language} medical text back to English.

Important: Translate literally and exactly. Do not correct any errors you perceive -
we need to see exactly what the text says.

Text to translate:
---
{translated_text}
---

Provide only the English translation, nothing else."""

# =============================================================================
# HUMAN EVALUATION SETTINGS
# =============================================================================

HUMAN_EVAL_CRITERIA = {
    "overall_accuracy": {
        "description": "How accurately does the translation convey the original meaning?",
        "scale": "1 (Very Poor) to 5 (Excellent)",
    },
    "medical_accuracy": {
        "description": "Are medical terms and concepts translated correctly?",
        "scale": "1 (Many errors) to 5 (Completely accurate)",
    },
    "cultural_appropriateness": {
        "description": "Is the translation culturally appropriate for the target audience?",
        "scale": "1 (Inappropriate) to 5 (Very appropriate)",
    },
    "clarity": {
        "description": "How easy is the translation to understand for a native speaker?",
        "scale": "1 (Very confusing) to 5 (Very clear)",
    },
    "safety_preservation": {
        "description": "Are all warnings.and safety information preserved?",
        "scale": "1 (Critical omissions) to 5 (Fully preserved)",
    },
    "actionability": {
        "description": "Can a patient follow the instructions based on this translation?",
        "scale": "1 (Cannot follow) to 5 (Easy to follow)",
    },
}

# =============================================================================
# EXECUTION SETTINGS
# =============================================================================

EXECUTION_CONFIG = {
    "batch_size": 5,  # Number of documents to process before saving checkpoint
    "retry_attempts": 3,  # Number of retries for failed API calls
    "retry_delay": 5,  # Seconds between retries
    "rate_limit_delay": 1,  # Seconds between API calls to avoid rate limits
    "save_intermediate": True,  # Save results after each document
    "parallel_models": False,  # Run models in parallel (use carefully - may hit rate limits)
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

import logging

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = OUTPUT_DIR / "translation_pipeline.log"

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("BackTranslation")
