# Back Translation Evaluation Scripts

## Quick Start

```bash
# 1. Install dependencies
pip install -r ../requirements.txt

# 2. Configure API keys in config.py or set environment variables:
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# 3. Verify setup
python run_pipeline.py --setup

# 4. Run quick test (one document, one language, one model)
python run_pipeline.py --test

# 5. Run full pipeline
python run_pipeline.py
```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `config.py` | Configuration, API keys, constants |
| `document_sources.py` | Patient education documents (12 samples included) |
| `translation_pipeline.py` | Translation via GPT-4, Claude, Gemini |
| `metrics_evaluation.py` | BLEU, ChrF, BERTScore, COMET, LaBSE |
| `results_aggregation.py` | Generate matrices, reports, visualizations |
| `human_evaluation.py` | Bilingual reviewer evaluation forms |
| `run_pipeline.py` | Main orchestrator |

## Pipeline Stages

```
┌─────────────────┐
│  Source Docs    │  12 patient education documents
│  (NIH/NLM)      │  8 categories: cardiology, diabetes, etc.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Translation    │  English → Target Language
│  Pipeline       │  Target Language → English (back-translation)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Automated      │  BLEU, ChrF, BERTScore
│  Metrics        │  COMET, LaBSE, Readability
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Aggregation    │  Model × Language matrices
│  & Reports      │  Kevin's scorecard format
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Human Eval     │  Bilingual reviewer packets
│  (Optional)     │  HEMAT/PMAT style scoring
└─────────────────┘
```

## Command Reference

```bash
# Full pipeline (all steps)
python run_pipeline.py

# Individual stages
python run_pipeline.py --setup        # Verify dependencies & API keys
python run_pipeline.py --test         # Quick test (1 doc, 1 lang, 1 model)
python run_pipeline.py --translate    # Run translations only
python run_pipeline.py --evaluate     # Calculate metrics only
python run_pipeline.py --report       # Generate reports only
python run_pipeline.py --human-eval   # Generate reviewer packets

# Test individual modules
python translation_pipeline.py --verify           # Check API connections
python translation_pipeline.py --test gpt-4-turbo spanish
python metrics_evaluation.py --test               # Test metric calculation
python document_sources.py                        # Show document stats
```

## Output Files

After running the pipeline:

```
output/
├── translations/
│   ├── all_results.json          # All translation results
│   └── checkpoint.json           # Resume checkpoint
├── metrics/
│   └── all_metrics.json          # All calculated metrics
└── reports/
    ├── kevin_scorecard.txt       # Main deliverable for Kevin
    ├── full_report.json          # Complete JSON report
    ├── translation_report.xlsx   # Excel workbook
    ├── translation_report.html   # HTML report
    ├── translation_report.csv    # CSV summary
    ├── heatmap_composite.png     # Visualization
    ├── model_comparison.png      # Model rankings
    ├── language_comparison.png   # Language rankings
    └── eval_packet_*.html        # Bilingual reviewer packets
```

## Configuration

Edit `config.py` to customize:

### Active Models
```python
ACTIVE_MODELS = [
    "gpt-4-turbo",
    "gpt-4o",
    "claude-3.5-sonnet",
    "gemini-1.5-pro",
]
```

### Active Languages
```python
ACTIVE_LANGUAGES = [
    "spanish",
    "chinese_simplified",
    "vietnamese",
    "tagalog",
    "russian",
    "arabic",
    "korean",
    "haitian_creole",  # Key test for underrepresented languages
]
```

### Metric Thresholds
```python
SUITABILITY_THRESHOLDS = {
    "suitable": 0.80,      # Green
    "caution": 0.60,       # Yellow
    "not_recommended": 0,  # Red
}
```

## Adding Custom Documents

```python
from document_sources import add_custom_document, save_documents_to_json

new_doc = add_custom_document(
    doc_id="CUSTOM_001",
    title="My Custom Document",
    category="cardiology",
    topic="custom_topic",
    source="My Hospital",
    english_text="Full text of the document..."
)

# Add to existing documents
from document_sources import load_documents_from_json
docs = load_documents_from_json()
docs.append(new_doc)
save_documents_to_json(docs)
```

## Bilingual Reviewer Workflow

1. Generate evaluation packets:
   ```bash
   python human_evaluation.py --generate
   ```

2. Send `eval_packet_[language].html` files to reviewers

3. Reviewers can also use Excel sheets: `eval_sheet_[language].xlsx`

4. Collect completed evaluations

5. Aggregate scores:
   ```bash
   python human_evaluation.py --aggregate reviewer1.xlsx reviewer2.xlsx
   ```

## Troubleshooting

### API Rate Limits
- The pipeline includes automatic delays between calls
- Adjust `EXECUTION_CONFIG["rate_limit_delay"]` in config.py if needed

### Missing COMET
- COMET requires large model downloads (~2GB)
- It's optional - pipeline works without it
- Install with: `pip install unbabel-comet`

### Memory Issues
- BERTScore and COMET load large models
- Reduce batch size if needed
- Run on machine with 16GB+ RAM recommended

### Resume After Interruption
- The pipeline saves checkpoints automatically
- Re-run `python run_pipeline.py --translate` to resume

## Metrics Reference

| Metric | Range | Excellent | What it Measures |
|--------|-------|-----------|------------------|
| BLEU | 0-100 | ≥40 | Lexical overlap (n-gram) |
| ChrF | 0-100 | ≥60 | Character-level similarity |
| BERTScore | 0-1 | ≥0.90 | Semantic similarity |
| COMET | 0-1 | ≥0.80 | Human judgment correlation |
| LaBSE | 0-1 | ≥0.85 | Cross-lingual similarity |
| Composite | 0-1 | ≥0.80 | Weighted combination |
