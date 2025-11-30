# Stanford Back Translation Evaluation Framework

A comprehensive framework for evaluating LLM translation quality of patient education materials using back-translation methodology.

## Overview

This project evaluates how well frontier Large Language Models (LLMs) translate clinical patient education documents. Using a **back-translation approach**, we translate documents from English to a target language, then back to English, and measure how much meaning is preserved.

### Key Results

| Model | Composite Score | COMET | BERTScore |
|-------|-----------------|-------|-----------|
| **Claude Opus 4.5** | **0.885** | **0.911** | **0.973** |
| GPT-5.1 | 0.869 | 0.903 | 0.951 |
| Gemini 3 Pro | 0.850 | 0.898 | 0.941 |
| Kimi K2 Thinking | 0.843 | 0.881 | 0.950 |

**Finding:** All models achieve "suitable" quality (>0.80), with Claude Opus 4.5 performing best across all metrics.

## Study Design

### Documents (12 total)
Patient education materials from MedlinePlus covering:

| Category | Documents |
|----------|-----------|
| Cardiology | Heart Failure Discharge, Taking Warfarin, Heart Attack Discharge, High Blood Pressure Q&A |
| Diabetes | Type 2 Diabetes Q&A, Low Blood Sugar Self-Care |
| Respiratory | Asthma Quick-Relief Drugs, COPD Q&A |
| Medication | ACE Inhibitors |
| Emergency | Stroke Warning Signs |
| Surgery | Knee Replacement Discharge, Hip Replacement Discharge |

### Languages (8 total)
- Spanish
- Chinese (Simplified)
- Vietnamese
- Tagalog/Filipino
- Russian
- Arabic
- Korean
- Haitian Creole

### Models (4 total)
1. **GPT-5.1** (OpenAI)
2. **Claude Opus 4.5** (Anthropic)
3. **Gemini 3 Pro** (Google)
4. **Kimi K2 Thinking** (Moonshot)

### Total Evaluations
- 12 documents × 8 languages × 4 models = **384 translations**

## Methodology

### Back-Translation Process
```
English Original → [LLM] → Target Language → [LLM] → English Back-Translation
                                                              ↓
                                                    Compare with Original
```

### Evaluation Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| **COMET** | 25% | Neural metric trained on human judgments - best predictor of human quality ratings |
| **BERTScore** | 25% | Semantic similarity using contextual embeddings |
| **LaBSE** | 20% | Cross-lingual semantic similarity |
| **ChrF** | 15% | Character-level F-score - good for morphologically rich languages |
| **BLEU** | 15% | Word n-gram overlap - traditional MT metric |

### Score Interpretation
- **≥0.85**: "Suitable" - High quality translation
- **0.75-0.85**: "Caution" - Review recommended
- **<0.75**: "Concern" - Significant issues likely

## Results by Language

| Language | Avg Composite | Notes |
|----------|---------------|-------|
| Spanish | 0.896 | Highest scores across all models |
| Tagalog | 0.884 | Strong performance |
| Haitian Creole | 0.873 | Good quality |
| Arabic | 0.871 | Good quality |
| Vietnamese | 0.858 | Good quality |
| Chinese (Simplified) | 0.844 | Lower due to character-level metric bias |
| Russian | 0.838 | Good quality |
| Korean | 0.830 | Lower due to character-level metric bias |

**Note:** Chinese and Korean scores appear lower partly due to character-level metrics (BLEU, ChrF) not accounting for the information density of CJK characters.

## Project Structure

```
Back translation project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── scripts/
│   ├── config.py                      # API keys and model configuration
│   ├── run_pipeline.py                # Main orchestration script
│   ├── translation_pipeline.py        # Translation logic
│   ├── metrics_evaluation.py          # Evaluation metrics
│   ├── results_aggregation.py         # Report generation
│   ├── document_sources.py            # Document management
│   └── human_evaluation.py            # Human eval materials
├── data/
│   └── source_documents/
│       └── source_documents.json      # 12 patient education documents
└── output/
    ├── translations/
    │   └── all_results.json           # All 384 translation pairs
    ├── metrics/
    │   └── all_metrics.json           # Complete metric scores
    └── reports/
        ├── translation_report.html    # Interactive visual report
        ├── translation_report.csv     # Excel-friendly export
        ├── kevin_scorecard.txt        # Quick summary
        ├── full_report.json           # Complete data
        └── EXECUTIVE_SUMMARY.md       # Executive summary
```

## Output Files

### For Quick Review
- **[translation_report.html](output/reports/translation_report.html)** - Interactive visual report with charts
- **[EXECUTIVE_SUMMARY.md](output/reports/EXECUTIVE_SUMMARY.md)** - High-level findings
- **[kevin_scorecard.txt](output/reports/kevin_scorecard.txt)** - Text scorecard

### For Data Analysis
- **[translation_report.csv](output/reports/translation_report.csv)** - All metrics in spreadsheet format
- **[all_metrics.json](output/metrics/all_metrics.json)** - Complete metric scores (JSON)
- **[all_results.json](output/translations/all_results.json)** - All 384 translation pairs

## Installation & Replication

### Prerequisites
- Python 3.9+
- API keys for: OpenAI, Anthropic, Google AI, Moonshot

### Setup
```bash
# Clone the repository
git clone [repo-url]
cd "Back translation project"

# Create virtual environment
python -m venv backtranslation_env
source backtranslation_env/bin/activate  # On Windows: backtranslation_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
# Edit scripts/config.py with your API keys
```

### Running the Pipeline
```bash
cd scripts

# Run full pipeline (translations + evaluation + reports)
python run_pipeline.py

# Or run individual steps:
python run_pipeline.py --translate   # Run translations only
python run_pipeline.py --evaluate    # Run metrics evaluation
python run_pipeline.py --report      # Generate reports
```

## Key Findings

1. **Claude Opus 4.5 leads** with the highest composite (0.885) and COMET (0.911) scores
2. **All models are production-viable** - every model exceeds the 0.80 "suitable" threshold
3. **Spanish translations excel** - highest scores across all models, likely due to training data abundance
4. **Back-translation is scalable** - enables quality assessment without native speakers for every document
5. **CJK metrics need interpretation** - Chinese/Korean scores reflect metric limitations, not necessarily quality issues

## Limitations

- Automated metrics correlate with but don't replace human judgment
- Back-translation may miss cultural/contextual nuances
- Results specific to patient education genre
- API costs can be significant (~$50-100 for full evaluation)

## Team

Stanford Clinical Translation Research Group

## License

[Add license]

---

*Generated: November 2024*
